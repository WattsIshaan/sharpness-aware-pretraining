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

def upload_to_gcs(local_dir: Path, gcs_path: str, dataset_name: str) -> None:
    """Upload directory contents to Google Cloud Storage."""
    if not gcs_path.startswith("gs://"):
        gcs_path = f"gs://{gcs_path}/"
    gcs_path = f"{gcs_path}/{dataset_name}"
    try:
        cmd = ["gsutil", "-m", "cp", "-r", f"{local_dir}/*", gcs_path]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info(f"Upload successful to: {gcs_path}")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to upload to GCS: {e.stderr}")
        raise

def load_dataset_with_fallback(dataset_name: str, config_name: str = None, split: str = "train"):
    """Load dataset with fallback to parquet revision."""
    try:
        if config_name:
            return ds.load_dataset(dataset_name, config_name, split=split)
        else:
            if dataset_name == "bigcode/starcoderdata":
                return ds.load_dataset(dataset_name, data_dir="python", split=split)
            return ds.load_dataset(dataset_name, split=split)
    except Exception as e:
        if "Dataset scripts are no longer supported" in str(e):
            return ds.load_dataset(dataset_name, config_name, split=split, revision="refs/convert/parquet") if config_name else ds.load_dataset(dataset_name, split=split, revision="refs/convert/parquet")
        raise

def preprocess(example, tokenizer: Tokenizer):
    """
    Modified Preprocess:
    1. Retains all specific format logic.
    2. Removed internal truncation/padding to allow for external packing.
    3. Ensures every example ends with a document separator (EOS).
    """
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    # --- RTE (Recognizing Textual Entailment) format ---
    if "text1" in example and "text2" in example and "label_text" in example:
        text1, text2, label_text = str(example["text1"]), str(example["text2"]), str(example["label_text"])
        prompt_text = f"<|user|>\nPremise: {text1}\nHypothesis: {text2}\nDoes the premise entail the hypothesis?\n"
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids += prompt_tokens
        label_mask += [False] * len(prompt_tokens)
        answer_text = f"<|assistant|>\n{label_text}{tokenizer.eos_token}\n"
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        input_ids += answer_tokens
        label_mask += [True] * len(answer_tokens)
    
    # --- TinyGSM format ---
    elif "question" in example and "code" in example:
        q, code = str(example["question"]), str(example["code"])
        q_tokens = tokenizer.encode(f"<|user|>\n{q}\n", add_special_tokens=False)
        input_ids += q_tokens
        label_mask += [False] * len(q_tokens)
        c_text = f"<|assistant|>\n{code}{tokenizer.eos_token}\n"
        c_tokens = tokenizer.encode(c_text, add_special_tokens=False)
        input_ids += c_tokens
        label_mask += [True] * len(c_tokens)
    
    # --- Multiple-choice QA format (SIQA, etc.) ---
    elif "question" in example and "label" in example and any(key in example for key in ["answerA", "answerB", "answerC"]):
        q, label = str(example["question"]), example["label"]
        # Choice selection
        if isinstance(label, str):
            l_str = label.strip().upper()
            answer = example.get("answerA" if l_str in ["A", "1"] else "answerB" if l_str in ["B", "2"] else "answerC", "")
        else:
            idx = int(label) - 1 if int(label) > 0 else int(label) # Handle 1-indexed vs 0-indexed
            answer = [example.get("answerA", ""), example.get("answerB", ""), example.get("answerC", "")][idx]
        
        ctx = str(example.get("context", "")).strip()
        prompt = f"<|user|>\n{ctx}\n{q}\n" if ctx else f"<|user|>\n{q}\n"
        p_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids += p_tokens
        label_mask += [False] * len(p_tokens)
        a_text = f"<|assistant|>\n{answer}{tokenizer.eos_token}\n"
        a_tokens = tokenizer.encode(a_text, add_special_tokens=False)
        input_ids += a_tokens
        label_mask += [True] * len(a_tokens)

    # --- Chat/Messages format ---
    elif "messages" in example:
        for msg in example["messages"]:
            role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
            input_ids += role_tokens
            label_mask += [False] * len(role_tokens)
            if msg["role"] == "assistant":
                content_tokens = tokenizer.encode(msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False)
                input_ids += content_tokens
                label_mask += [True] * len(content_tokens)
            else:
                content_tokens = tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
                input_ids += content_tokens
                label_mask += [False] * len(content_tokens)

    # --- Alpaca format ---
    elif "instruction" in example:
        instr, inp, out = str(example["instruction"]), str(example.get("input", "")), str(example.get("output", ""))
        prompt = f"<|user|>\n{instr}\n{inp}\n" if inp else f"<|user|>\n{instr}\n"
        p_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids += p_tokens
        label_mask += [False] * len(p_tokens)
        a_tokens = tokenizer.encode(f"<|assistant|>\n{out}{tokenizer.eos_token}\n", add_special_tokens=False)
        input_ids += a_tokens
        label_mask += [True] * len(a_tokens)

    # --- Raw text format ---
    elif "text" in example:
        tokens = tokenizer.encode(str(example["text"]) + tokenizer.eos_token, add_special_tokens=False)
        input_ids += tokens
        label_mask += [True] * len(tokens)

    # --- Prompt-Completion or Standard QA ---
    elif ("prompt" in example and "completion" in example) or ("question" in example and "answer" in example):
        p = str(example.get("prompt", example.get("question", "")))
        a = str(example.get("completion", example.get("answer", "")))
        p_tokens = tokenizer.encode(f"<|user|>\n{p}\n", add_special_tokens=False)
        a_tokens = tokenizer.encode(f"<|assistant|>\n{a}{tokenizer.eos_token}\n", add_special_tokens=False)
        input_ids += p_tokens + a_tokens
        label_mask += [False] * len(p_tokens) + [True] * len(a_tokens)
    
    else:
        # Final fallback: search for any text-like field
        for field in ["content", "document", "code", "solution"]:
            if field in example:
                text = str(example[field]).strip()
                tokens = tokenizer.encode(text + tokenizer.eos_token, add_special_tokens=False)
                input_ids += tokens
                label_mask += [True] * len(tokens)
                break

    # Final label mask cleanup and EOS document separation
    if len(label_mask) > 0:
        label_mask[-1] = False # Avoid predicting the very last newline/EOS uniquely
    
    if len(input_ids) > 0 and input_ids[-1] != tokenizer.eos_token_id:
        input_ids.append(tokenizer.eos_token_id)
        label_mask.append(False)

    return {
        "input_ids": input_ids,
        "label_mask": label_mask,
        "n_labels": sum(label_mask),
        "n_tokens": len(input_ids),
    }

def main(opts) -> None:
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    dtype = np.uint32 if str(opts.tokenizer).endswith("allenai_dolma2.json") else np.uint16
    
    train_datasets = []
    val_datasets = []
    for d_spec in opts.datasets:
        name, config = d_spec.split(":", 1) if ":" in d_spec else (d_spec, None)
        log.info(f"Loading {name}...")
        train_datasets.append(load_dataset_with_fallback(name, config, split="train"))
        for split_name in ("validation", "test"):
            try:
                val_datasets.append(load_dataset_with_fallback(name, config, split=split_name))
                log.info(f"Using '{split_name}' split for validation from {name}.")
                break
            except Exception:
                continue
    
    train_dataset = (
        ds.concatenate_datasets(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    )
    val_dataset = (
        ds.concatenate_datasets(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        if val_datasets
        else None
    )
    
    log.info("Tokenizing variable length sequences...")
    train_dataset = train_dataset.map(
        partial(preprocess, tokenizer=tokenizer),
        batched=False,
        remove_columns=train_dataset.column_names,
        num_proc=opts.num_proc,
    )
    train_dataset = train_dataset.filter(lambda x: x["n_labels"] > 0, num_proc=opts.num_proc)
    if val_dataset is not None:
        val_dataset = val_dataset.map(
            partial(preprocess, tokenizer=tokenizer),
            batched=False,
            remove_columns=val_dataset.column_names,
            num_proc=opts.num_proc,
        )
        val_dataset = val_dataset.filter(lambda x: x["n_labels"] > 0, num_proc=opts.num_proc)

    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)

    train_total_tokens = int(np.sum(train_dataset["n_tokens"]))
    val_total_tokens = int(np.sum(val_dataset["n_tokens"])) if val_dataset is not None else 0
    total_available = train_total_tokens + val_total_tokens
    max_tokens = min(opts.max_tokens, total_available if val_dataset is not None else train_total_tokens)
    log.info(f"Dataset length (train docs): {len(train_dataset)}")
    if val_dataset is not None:
        log.info(f"Dataset length (val docs): {len(val_dataset)}")

    # 80/20 Split based on token blocks, bounded by available tokens
    train_target = min(int(max_tokens * 0.8), train_total_tokens)
    val_target = (
        min(int(max_tokens * 0.2), val_total_tokens)
        if val_dataset is not None
        else min(int(max_tokens * 0.2), max(0, train_total_tokens - train_target))
    )
    train_tokens = (train_target // opts.seq_len) * opts.seq_len
    val_tokens = (val_target // opts.seq_len) * opts.seq_len
    total_target = train_tokens + val_tokens

    train_ids = np.memmap(str(output_dir / "train/input_ids.npy"), dtype=dtype, mode="w+", shape=(train_tokens,))
    train_mask = np.memmap(str(output_dir / "train/label_mask.npy"), dtype=np.bool_, mode="w+", shape=(train_tokens,))
    val_ids = np.memmap(str(output_dir / "val/input_ids.npy"), dtype=dtype, mode="w+", shape=(val_tokens,))
    val_mask = np.memmap(str(output_dir / "val/label_mask.npy"), dtype=np.bool_, mode="w+", shape=(val_tokens,))

    log.info(
        f"Packing {total_target:,d} tokens into memmap (Train: {train_tokens:,d}, Val: {val_tokens:,d})..."
    )
    
    train_tokens_written = 0
    val_tokens_written = 0
    train_examples_saved = 0
    val_examples_saved = 0
    if val_dataset is None:
        curr_tokens = 0
        for ex in track(train_dataset, description="Writing to disk"):
            ids, mask = ex["input_ids"], ex["label_mask"]
            n = len(ids)
            if curr_tokens < train_tokens:
                chunk = min(n, train_tokens - curr_tokens)
                train_ids[curr_tokens : curr_tokens + chunk] = ids[:chunk]
                train_mask[curr_tokens : curr_tokens + chunk] = mask[:chunk]
                train_tokens_written += chunk
                train_examples_saved += 1
            elif curr_tokens < total_target:
                v_offset = curr_tokens - train_tokens
                chunk = min(n, total_target - curr_tokens)
                val_ids[v_offset : v_offset + chunk] = ids[:chunk]
                val_mask[v_offset : v_offset + chunk] = mask[:chunk]
                val_tokens_written += chunk
                val_examples_saved += 1
            else:
                break
            curr_tokens += n
    else:
        for ex in track(train_dataset, description="Writing train to disk"):
            ids, mask = ex["input_ids"], ex["label_mask"]
            n = len(ids)
            if train_tokens_written >= train_tokens:
                break
            chunk = min(n, train_tokens - train_tokens_written)
            train_ids[train_tokens_written : train_tokens_written + chunk] = ids[:chunk]
            train_mask[train_tokens_written : train_tokens_written + chunk] = mask[:chunk]
            train_tokens_written += chunk
            train_examples_saved += 1

        if val_tokens > 0:
            for ex in track(val_dataset, description="Writing val to disk"):
                ids, mask = ex["input_ids"], ex["label_mask"]
                n = len(ids)
                if val_tokens_written >= val_tokens:
                    break
                chunk = min(n, val_tokens - val_tokens_written)
                val_ids[val_tokens_written : val_tokens_written + chunk] = ids[:chunk]
                val_mask[val_tokens_written : val_tokens_written + chunk] = mask[:chunk]
                val_tokens_written += chunk
                val_examples_saved += 1

    train_ids.flush()
    val_ids.flush()
    log.info("--- Processing Summary ---")
    log.info(f"Train tokens saved: {train_tokens_written:,d}")
    log.info(f"Val tokens saved: {val_tokens_written:,d}")
    log.info(
        f"Train/Val blocks ({opts.seq_len}): {train_tokens//opts.seq_len} / {val_tokens//opts.seq_len}"
    )
    log.info(
        f"Train/Val examples packed: {train_examples_saved:,d} / {val_examples_saved:,d}"
    )

    if opts.gcs_bucket:
        assert len(opts.datasets) == 1, "Only one dataset can be uploaded to GCS at a time"
        dataset_name = opts.datasets[0].split(":")[0].replace("/", "_")
        upload_to_gcs(output_dir, opts.gcs_bucket, dataset_name)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-d", "--datasets", nargs='+', required=True)
    parser.add_argument("-t", "--tokenizer", default="allenai/eleuther-ai-gpt-neox-20b-pii-special")
    parser.add_argument("-s", "--seq-len", type=int, default=1024)
    parser.add_argument("--eos", type=int, default=0)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("-j", "--num-proc", type=int, default=8)
    parser.add_argument("-m", "--max-tokens", type=int, default=100_000_000)
    parser.add_argument("-g", "--gcs-bucket", default=None)
    return parser

if __name__ == "__main__":
    prepare_cli_environment()
    main(get_parser().parse_args())