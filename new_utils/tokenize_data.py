"""
Script for preparing dataset(s) for fine-tuning an OLMo model.
Supports tokenizing one or multiple datasets from the command line.
Handles multiple dataset formats (chat, instruction, QA, etc.).
Supports dataset configs with 'dataset:config' syntax.
Splits data 80/20 into train and val directories.
Optionally uploads tokenized data to Google Cloud Storage.
"""

import logging
import subprocess
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track

from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


def upload_to_gcs(local_dir: Path, gcs_path: str) -> None:
    """Upload directory contents to Google Cloud Storage."""
    log.info(f"Uploading to GCS: {gcs_path}")
    
    # Ensure gcs_path starts with gs://
    if not gcs_path.startswith("gs://"):
        gcs_path = f"gs://{gcs_path}"
    
    try:
        # Use gsutil to copy the directory recursively
        cmd = ["gsutil", "-m", "cp", "-r", f"{local_dir}/*", gcs_path]
        log.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info(f"Upload successful! Data available at: {gcs_path}")
        if result.stdout:
            log.info(result.stdout)
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to upload to GCS: {e}")
        log.error(f"stderr: {e.stderr}")
        raise


def load_dataset_with_fallback(dataset_name: str, config_name: str = None, split: str = "train"):
    """Load dataset with fallback to parquet revision if dataset scripts are not supported."""
    try:
        if config_name:
            return ds.load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
        else:
            return ds.load_dataset(dataset_name, split=split, trust_remote_code=True)
    except RuntimeError as e:
        # Only retry with parquet if the error is about dataset scripts not being supported
        if "Dataset scripts are no longer supported" in str(e):
            log.warning(f"Dataset scripts not supported: {e}")
            log.info("Retrying with revision='refs/convert/parquet'...")
            try:
                if config_name:
                    return ds.load_dataset(dataset_name, config_name, split=split, trust_remote_code=True, revision="refs/convert/parquet")
                else:
                    return ds.load_dataset(dataset_name, split=split, trust_remote_code=True, revision="refs/convert/parquet")
            except Exception as e2:
                log.error(f"Failed to load with parquet revision: {e2}")
                raise
        else:
            # Re-raise if it's a different RuntimeError
            raise


def main(opts) -> None:
    tokenizer: Tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    
    # Set dtype depending on tokenizer
    # If tokenizer path ends with "tokenizers/allenai_dolma2.json", use uint32; else use uint16
    dtype = np.uint32 if (
        isinstance(opts.tokenizer, str) and opts.tokenizer.endswith("tokenizers/allenai_dolma2.json")
    ) else np.uint16
    # Load and concatenate all datasets
    all_datasets = []
    for dataset_spec in opts.datasets:
        # Parse dataset spec: format can be "name" or "name:config"
        if ":" in dataset_spec:
            dataset_name, config_name = dataset_spec.split(":", 1)
            log.info(f"Loading dataset: {dataset_name} (config: {config_name})")
            try:
                dataset = load_dataset_with_fallback(dataset_name, config_name)
            except Exception as e:
                log.error(f"Failed to load {dataset_name} with config {config_name}: {e}")
                log.info("Trying without config...")
                dataset = load_dataset_with_fallback(dataset_name)
        else:
            dataset_name = dataset_spec
            log.info(f"Loading dataset: {dataset_name}")
            try:
                dataset = load_dataset_with_fallback(dataset_name)
            except ValueError as e:
                # Check if it's a missing config error
                if "Config name is missing" in str(e):
                    log.info("Dataset requires a config. Trying with 'main' config...")
                    dataset = load_dataset_with_fallback(dataset_name, "main")
                else:
                    raise
        all_datasets.append(dataset)
    
    # Concatenate all datasets
    if len(all_datasets) > 1:
        log.info(f"Concatenating {len(all_datasets)} datasets...")
        dataset = ds.concatenate_datasets(all_datasets)
    else:
        dataset = all_datasets[0]
    
    log.info(f"Total examples: {len(dataset):,d}")

    # Detect dataset format and prepare preprocessing
    sample = dataset[0]
    log.info(f"Dataset columns: {list(sample.keys())}")
    
    # Log a sample to help understand the format
    if "text1" in sample and "text2" in sample and "label_text" in sample:
        log.info("Detected RTE (Recognizing Textual Entailment) format")
    elif "question" in sample and "code" in sample:
        log.info("Detected TinyGSM (question -> code) format")
    elif "question" in sample and "label" in sample and any(key in sample for key in ["answerA", "answerB", "answerC"]):
        log.info("Detected multiple-choice QA format (like SIQA)")
    elif "messages" in sample:
        log.info("Detected chat/messages format")
    elif "instruction" in sample:
        log.info("Detected Alpaca-style instruction format")
    elif "question" in sample and "answer" in sample:
        log.info("Detected QA format")
    elif "text" in sample:
        log.info("Detected raw text format")
    elif "prompt" in sample and "completion" in sample:
        log.info("Detected prompt-completion format")
    else:
        log.warning(f"Unknown format, will attempt to process. Sample keys: {list(sample.keys())}")
    
    # Determine which columns to remove (keep only input_ids, label_mask, n_labels after preprocessing)
    original_columns = list(sample.keys())
    
    max_tokens = opts.max_tokens
    
    # Estimate how many examples we need with buffer for filtering
    buffer_multiplier = opts.buffer_multiplier
    estimated_examples_needed = int((max_tokens / opts.seq_len) * buffer_multiplier)
    
    # Limit to available data
    total_examples_available = len(dataset)  # type: ignore
    examples_to_process = min(estimated_examples_needed, total_examples_available)
    
    log.info(f"Target tokens: {max_tokens:,d}")
    log.info(f"Estimated examples needed: {estimated_examples_needed:,d} (with {int((buffer_multiplier-1)*100)}% buffer for filtering)")
    log.info(f"Will process: {examples_to_process:,d} examples out of {total_examples_available:,d} available")
    
    # Select subset to process
    if examples_to_process < total_examples_available:
        dataset = dataset.select(range(examples_to_process))  # type: ignore
        log.info(f"Selected first {examples_to_process:,d} examples for processing")
    
    log.info("Tokenizing dataset...")
    dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
        batched=False,
        remove_columns=original_columns,  # Remove all original columns
        num_proc=opts.num_proc,  # type: ignore
    )

    log.info("Filtering dataset...")
    n = len(dataset)  # type: ignore
    dataset = dataset.filter(filter, batched=False, num_proc=opts.num_proc)  # type: ignore
    log.info(f"Filtered out {n - len(dataset):,d} examples ({(n - len(dataset))/n*100:.1f}%)")

    log.info(f"Collecting tokens (max limit: {max_tokens:,d} tokens)...")
    total_tokens = 0
    examples_to_keep = []
    
    for ex in track(dataset):
        assert len(ex["input_ids"]) == opts.seq_len  # type: ignore
        ex_tokens = len(ex["input_ids"])  # type: ignore
        
        # Check if adding this example would exceed the limit
        if total_tokens + ex_tokens > max_tokens:
            # Only add partial tokens to reach exactly max_tokens
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 0:
                examples_to_keep.append((ex, remaining_tokens))
                total_tokens = max_tokens
            break
        else:
            examples_to_keep.append((ex, ex_tokens))
            total_tokens += ex_tokens
    
    if total_tokens == max_tokens:
        log.info(f"✓ Total tokens to save: {total_tokens:,d} (reached max limit)")
    elif total_tokens >= max_tokens * 0.95:
        log.info(f"✓ Total tokens to save: {total_tokens:,d} ({total_tokens/max_tokens*100:.1f}% of target)")
    else:
        log.warning(f"⚠ Total tokens to save: {total_tokens:,d} ({total_tokens/max_tokens*100:.1f}% of target - consider increasing buffer or dataset size)")
    log.info(f"Number of examples: {len(examples_to_keep):,d}")

    # Split ~80/20 for train/val at example boundaries
    # Train ends at seq_len boundary, val starts with complete example
    train_split = 0.8
    num_train_examples = int(len(examples_to_keep) * train_split)
    
    # Ensure we don't include partial examples in the split
    train_examples = []
    val_examples = []
    train_token_count = 0
    
    for idx, (ex, num_tokens) in enumerate(examples_to_keep):
        if idx < num_train_examples:
            # Add to train
            train_examples.append((ex, num_tokens))
            train_token_count += num_tokens
        else:
            # Add complete sequences to val (each example is seq_len)
            val_examples.append((ex, opts.seq_len))
    
    # Ensure train ends at seq_len boundary (multiple of context length)
    # Round down to nearest multiple
    train_tokens_rounded = (train_token_count // opts.seq_len) * opts.seq_len
    
    # If train tokens exceed the rounded boundary, move last examples to val
    if train_tokens_rounded < train_token_count:
        # Find where to cut train to end at seq_len boundary
        adjusted_train = []
        cumulative = 0
        for ex, num_tokens in train_examples:
            if cumulative + num_tokens <= train_tokens_rounded:
                adjusted_train.append((ex, num_tokens))
                cumulative += num_tokens
            else:
                # Move this and remaining examples to val
                val_examples.insert(0, (ex, opts.seq_len))
        train_examples = adjusted_train
        train_tokens = train_tokens_rounded
    else:
        train_tokens = train_token_count
    
    val_tokens = len(val_examples) * opts.seq_len
    
    log.info(f"Splitting data: {train_tokens:,d} train tokens ({train_tokens/total_tokens*100:.1f}%), {val_tokens:,d} val tokens ({val_tokens/total_tokens*100:.1f}%)")
    log.info(f"Train: {len(train_examples):,d} examples = {train_tokens // opts.seq_len} complete seq_len blocks")
    log.info(f"Val: {len(val_examples):,d} complete examples of length {opts.seq_len}")

    log.info(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create train and val subdirectories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True, parents=True)
    val_dir.mkdir(exist_ok=True, parents=True)

    # Create memory-mapped files for train
    train_input_ids = np.memmap(
        str(train_dir / "input_ids.npy"), dtype=dtype, mode="w+", shape=(train_tokens,)
    )
    train_label_mask = np.memmap(
        str(train_dir / "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(train_tokens,)
    )
    
    # Create memory-mapped files for val (complete sequences only)
    val_input_ids = np.memmap(
        str(val_dir / "input_ids.npy"), dtype=dtype, mode="w+", shape=(val_tokens,)
    )
    val_label_mask = np.memmap(
        str(val_dir / "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(val_tokens,)
    )
    
    # Write train data
    log.info("Writing train data...")
    train_offset = 0
    for ex, num_tokens in track(train_examples, description="Writing train"):
        train_input_ids[train_offset : train_offset + num_tokens] = ex["input_ids"][:num_tokens]  # type: ignore
        train_label_mask[train_offset : train_offset + num_tokens] = ex["label_mask"][:num_tokens]  # type: ignore
        train_offset += num_tokens
    
    # Write val data (complete sequences)
    log.info("Writing val data...")
    val_offset = 0
    for ex, _ in track(val_examples, description="Writing val"):
        # Each val example is a complete sequence of seq_len
        val_input_ids[val_offset : val_offset + opts.seq_len] = ex["input_ids"][:opts.seq_len]  # type: ignore
        val_label_mask[val_offset : val_offset + opts.seq_len] = ex["label_mask"][:opts.seq_len]  # type: ignore
        val_offset += opts.seq_len
    
    log.info("Flushing train data...")
    train_input_ids.flush()
    train_label_mask.flush()
    
    log.info("Flushing val data...")
    val_input_ids.flush()
    val_label_mask.flush()
    
    log.info(f"Train data saved to: {train_dir}")
    log.info(f"Val data saved to: {val_dir}")

    # Upload to GCS if bucket path is provided
    if opts.gcs_bucket:
        # Create dataset name from the list of datasets
        if len(opts.datasets) == 1:
            dataset_name = opts.datasets[0].replace("/", "_").replace(":", "_")
        else:
            dataset_name = "_".join([d.replace("/", "_").replace(":", "_") for d in opts.datasets])
        
        gcs_path = f"{opts.gcs_bucket.rstrip('/')}/{dataset_name}"
        log.info(f"Uploading train/ and val/ directories to GCS...")
        upload_to_gcs(output_dir, gcs_path)
        log.info(f"Data uploaded to: {gcs_path}/train/ and {gcs_path}/val/")
    
    log.info("Done!")


def filter(example):
    return example["n_labels"] > 0


def preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    # Handle different dataset formats
    # Check RTE (Recognizing Textual Entailment) format FIRST
    if "text1" in example and "text2" in example and "label_text" in example:
        # RTE format: text1 (premise), text2 (hypothesis), label_text (entailment/not_entailment)
        text1 = example["text1"].strip() if isinstance(example["text1"], str) else str(example["text1"])
        text2 = example["text2"].strip() if isinstance(example["text2"], str) else str(example["text2"])
        label_text = example["label_text"].strip() if isinstance(example["label_text"], str) else str(example["label_text"])
        
        # Format: Premise: <text1>\n\nHypothesis: <text2>\n\nDoes the premise entail the hypothesis?
        prompt_text = f"<|user|>\nPremise: {text1}\n\nHypothesis: {text2}\n\nDoes the premise entail the hypothesis?\n"
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids += prompt_tokens
        label_mask += [False] * len(prompt_tokens)
        
        # Answer with the label
        answer_text = f"<|assistant|>\n{label_text}{tokenizer.eos_token}\n"
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        input_ids += answer_tokens
        label_mask += [True] * len(answer_tokens)
        # Don't predict the final newline
        if len(label_mask) > 0:
            label_mask[-1] = False
    
    # Check tinygsm format (question and code)
    elif "question" in example and "code" in example:
        # TinyGSM format: question (math problem), code (solution code)
        question = example["question"].strip() if isinstance(example["question"], str) else str(example["question"])
        code = example["code"].strip() if isinstance(example["code"], str) else str(example["code"])
        
        # Format the question as a prompt
        question_text = f"<|user|>\n{question}\n"
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        input_ids += question_tokens
        label_mask += [False] * len(question_tokens)  # Mask the question
        
        # Generate code as the response
        code_text = f"<|assistant|>\n{code}{tokenizer.eos_token}\n"
        code_tokens = tokenizer.encode(code_text, add_special_tokens=False)
        input_ids += code_tokens
        label_mask += [True] * len(code_tokens)  # Only compute loss on code
        # Don't predict the final newline
        if len(label_mask) > 0:
            label_mask[-1] = False
    
    # Check multiple-choice format (SIQA, etc.)
    elif "question" in example and "label" in example and any(key in example for key in ["answerA", "answerB", "answerC"]):
        # Multiple-choice QA format (like SIQA)
        question = example["question"].strip() if isinstance(example["question"], str) else str(example["question"])
        label = example["label"]
        
        # Map label to answer choice
        if isinstance(label, str):
            # If label is a string like "A", "B", "C" or "1", "2", "3"
            label_str = label.strip().upper()
            if label_str in ["A", "1"]:
                answer = example.get("answerA", "")
            elif label_str in ["B", "2"]:
                answer = example.get("answerB", "")
            elif label_str in ["C", "3"]:
                answer = example.get("answerC", "")
            else:
                answer = example.get("answerA", "")  # Default to A
        else:
            # If label is an integer (1, 2, 3 for SIQA which is 1-indexed)
            label_int = int(label)
            if label_int == 1:
                answer = example.get("answerA", "")
            elif label_int == 2:
                answer = example.get("answerB", "")
            elif label_int == 3:
                answer = example.get("answerC", "")
            else:
                # For 0-indexed labels
                answers = [example.get("answerA", ""), example.get("answerB", ""), example.get("answerC", "")]
                if 0 <= label_int < len(answers):
                    answer = answers[label_int]
                else:
                    answer = answers[0] if answers else ""
        
        answer = answer.strip() if isinstance(answer, str) else str(answer)
        
        # Add context if available
        context = example.get("context", "")
        if context and context.strip():
            context_text = context.strip() if isinstance(context, str) else str(context)
            question_text = f"<|user|>\n{context_text}\n{question}\n"
        else:
            question_text = f"<|user|>\n{question}\n"
        
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        input_ids += question_tokens
        label_mask += [False] * len(question_tokens)
        
        answer_text = f"<|assistant|>\n{answer}{tokenizer.eos_token}\n"
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        input_ids += answer_tokens
        label_mask += [True] * len(answer_tokens)
        if len(label_mask) > 0:
            label_mask[-1] = False
    
    elif "messages" in example:
        # Chat/instruction format with messages (e.g., Tulu)
        for msg in example["messages"]:
            role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
            label_mask += [False] * len(role_tokens)
            input_ids += role_tokens

            if msg["role"] == "assistant":
                content_tokens = tokenizer.encode(
                    msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False
                )
                label_mask += [True] * len(content_tokens)
                # mask out the last '\n'
                if len(content_tokens) >= 2 and content_tokens[-2] == tokenizer.eos_token_id:
                    label_mask[-1] = False
            else:
                content_tokens = tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
                label_mask += [False] * len(content_tokens)
            input_ids += content_tokens
            
    elif "instruction" in example:
        # Alpaca-style format
        instruction = example["instruction"].strip()
        input_text = example.get("input", "").strip()
        output = example.get("output", "").strip()
        
        # Encode instruction
        if input_text:
            prompt = f"<|user|>\n{instruction}\n{input_text}\n"
        else:
            prompt = f"<|user|>\n{instruction}\n"
        
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids += prompt_tokens
        label_mask += [False] * len(prompt_tokens)
        
        # Encode output (this is what we want to predict)
        output_text = f"<|assistant|>\n{output}{tokenizer.eos_token}\n"
        output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
        input_ids += output_tokens
        label_mask += [True] * len(output_tokens)
        # Don't predict the final newline
        if len(label_mask) > 0:
            label_mask[-1] = False
            
    elif "text" in example:
        # Raw text format
        text = example["text"].strip()
        text_tokens = tokenizer.encode(text + tokenizer.eos_token, add_special_tokens=False)
        input_ids += text_tokens
        label_mask += [True] * len(text_tokens)
        
    elif "prompt" in example and "completion" in example:
        # Prompt-completion format
        prompt = example["prompt"].strip()
        completion = example["completion"].strip()
        
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids += prompt_tokens
        label_mask += [False] * len(prompt_tokens)
        
        completion_tokens = tokenizer.encode(completion + tokenizer.eos_token, add_special_tokens=False)
        input_ids += completion_tokens
        label_mask += [True] * len(completion_tokens)
        
    elif "question" in example and "answer" in example:
        # Regular QA format (like GSM8K with direct answer)
        question = example["question"].strip() if isinstance(example["question"], str) else str(example["question"])
        answer = example["answer"].strip() if isinstance(example.get("answer", ""), str) else str(example.get("answer", ""))
        
        # Add context if available
        context = example.get("context", "")
        if context and context.strip():
            context_text = context.strip() if isinstance(context, str) else str(context)
            question_text = f"<|user|>\n{context_text}\n{question}\n"
        else:
            question_text = f"<|user|>\n{question}\n"
        
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        input_ids += question_tokens
        label_mask += [False] * len(question_tokens)
        
        answer_text = f"<|assistant|>\n{answer}{tokenizer.eos_token}\n"
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        input_ids += answer_tokens
        label_mask += [True] * len(answer_tokens)
        if len(label_mask) > 0:
            label_mask[-1] = False
    else:
        # Fallback: try to find any text-like field
        log.warning(f"Unknown dataset format with keys: {list(example.keys())}")
        # Try common field names
        for field in ["content", "document", "code", "solution"]:
            if field in example:
                text = example[field].strip()
                text_tokens = tokenizer.encode(text + tokenizer.eos_token, add_special_tokens=False)
                input_ids += text_tokens
                label_mask += [True] * len(text_tokens)
                break

    # Truncate or pad to max_seq_len
    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare dataset(s) for fine-tuning")
    parser.add_argument("output_dir", type=str, help="""Directory to save the results to.""")
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        nargs='+',
        required=True,
        help="""One or more dataset names/paths to tokenize. 
                Format: 'dataset_name' or 'dataset_name:config_name'
                Examples: 'allenai/tulu-v2-sft-mixture', 'openai/gsm8k:main'""",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default=Path(__file__).parent / "tokenizers" / "allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
    )
    parser.add_argument("-s", "--seq-len", type=int, help="""Max sequence length.""", default=2048)
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=50279)
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=1)
    parser.add_argument("-j", "--num-proc", type=int, help="""Number of workers.""", default=8)
    parser.add_argument(
        "-m",
        "--max-tokens",
        type=int,
        help="""Maximum number of tokens to save (default: 100M).""",
        default=100_000_000,  # 100 million tokens by default
    )
    parser.add_argument(
        "-b",
        "--buffer-multiplier",
        type=float,
        help="""Buffer multiplier for early stopping (default: 1.4). 
                Processes (max_tokens / seq_len) * buffer_multiplier examples to account for filtering.
                Increase if many examples are filtered out.""",
        default=1.4,
    )
    parser.add_argument(
        "-g",
        "--gcs-bucket",
        type=str,
        help="""GCS bucket path to upload results (e.g., gs://my-bucket/path or my-bucket/path).""",
        default=None,
    )
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)