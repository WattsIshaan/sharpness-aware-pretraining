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
    if not gcs_path.startswith("gs://"):
        gcs_path = f"gs://{gcs_path}"
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

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": sum(label_mask)}

def main(opts) -> None:
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    dtype = np.uint32 if str(opts.tokenizer).endswith("allenai_dolma2.json") else np.uint16
    
    all_datasets = []
    for d_spec in opts.datasets:
        name, config = d_spec.split(":", 1) if ":" in d_spec else (d_spec, None)
        log.info(f"Loading {name}...")
        all_datasets.append(load_dataset_with_fallback(name, config))
    
    dataset = ds.concatenate_datasets(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
    
    log.info("Tokenizing variable length sequences...")
    dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=False, remove_columns=dataset.column_names, num_proc=opts.num_proc)
    dataset = dataset.filter(lambda x: x["n_labels"] > 0, num_proc=opts.num_proc)

    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)

    # 80/20 Split based on token blocks
    train_tokens = int((opts.max_tokens * 0.8) // opts.seq_len) * opts.seq_len
    val_tokens = int((opts.max_tokens * 0.2) // opts.seq_len) * opts.seq_len
    total_target = train_tokens + val_tokens

    train_ids = np.memmap(str(output_dir / "train/input_ids.npy"), dtype=dtype, mode="w+", shape=(train_tokens,))
    train_mask = np.memmap(str(output_dir / "train/label_mask.npy"), dtype=np.bool_, mode="w+", shape=(train_tokens,))
    val_ids = np.memmap(str(output_dir / "val/input_ids.npy"), dtype=dtype, mode="w+", shape=(val_tokens,))
    val_mask = np.memmap(str(output_dir / "val/label_mask.npy"), dtype=np.bool_, mode="w+", shape=(val_tokens,))

    log.info(f"Packing {total_target:,d} tokens into memmap (Train: {train_tokens:,d}, Val: {val_tokens:,d})...")
    
    curr_tokens = 0
    examples_saved = 0
    for ex in track(dataset, description="Writing to disk"):
        ids, mask = ex["input_ids"], ex["label_mask"]
        n = len(ids)
        
        if curr_tokens < train_tokens:
            chunk = min(n, train_tokens - curr_tokens)
            train_ids[curr_tokens : curr_tokens + chunk] = ids[:chunk]
            train_mask[curr_tokens : curr_tokens + chunk] = mask[:chunk]
        elif curr_tokens < total_target:
            v_offset = curr_tokens - train_tokens
            chunk = min(n, total_target - curr_tokens)
            val_ids[v_offset : v_offset + chunk] = ids[:chunk]
            val_mask[v_offset : v_offset + chunk] = mask[:chunk]
        else:
            break
        curr_tokens += n
        examples_saved += 1

    train_ids.flush()
    val_ids.flush()
    log.info("--- Processing Summary ---")
    log.info(f"Total tokens saved: {curr_tokens:,d}")
    log.info(f"Total examples (documents) packed: {examples_saved:,d}")
    log.info(f"Train/Val blocks ({opts.seq_len}): {train_tokens//opts.seq_len} / {val_tokens//opts.seq_len}")

    if opts.gcs_bucket:
        upload_to_gcs(output_dir, opts.gcs_bucket)

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