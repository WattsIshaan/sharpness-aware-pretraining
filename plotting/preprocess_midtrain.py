import json
import os
import re
from utils.config_globals import RESULTS_DIR


# ---------------------------------------------------------------------------
# Base Model eval tasks (keys as they appear in the JSONL files)
# ---------------------------------------------------------------------------

BASE_MCQ_TASKS = [
    "arc_challenge::olmes",
    "boolq::olmes",
    "hellaswag::olmes",
    "mmlu:mc::olmes",
    "winogrande::olmes",
]

BASE_GENERATIVE_TASKS = [
    "drop",
    "naturalqs_open",
]

BASE_HELDOUT_TASKS = [
    "agi_eval_english:1shot::olmes",
    "gsm8k",
    "mmlu_pro:mc::none",
    "triviaqa",
]

BASE_EVAL_GROUPS = {
    "mcq": BASE_MCQ_TASKS,
    "generative": BASE_GENERATIVE_TASKS,
    "heldout": BASE_HELDOUT_TASKS,
}

# ---------------------------------------------------------------------------
# SFT Model eval tasks (keys as they appear in the JSONL files)
# ---------------------------------------------------------------------------

SFT_KNOWLEDGE_TASKS = [
    "mmlu:mc::tulu",
    "popqa",
    "truthfulqa",
]

SFT_REASONING_TASKS = [
    "bbh:cot-v1::tulu",
    "minerva_math::tulu",
    "gsm8k",
    "drop",
]

SFT_CODE_TASKS = [
    "codex_humaneval",
    "codex_humanevalplus",
]

SFT_IF_TASKS = [
    "ifeval",
]

SFT_EVAL_GROUPS = {
    "knowledge": SFT_KNOWLEDGE_TASKS,
    "reasoning": SFT_REASONING_TASKS,
    "code": SFT_CODE_TASKS,
    "instruction_following": SFT_IF_TASKS,
}


def _clean_task_name(task_name: str) -> str:
    """Strip everything after the first ':' from a task name."""
    return task_name.split(":")[0]


def _group_task_metrics(raw_task_metrics: dict, eval_groups: dict) -> dict:
    """Organise raw task_name → score dict into headline groups."""
    grouped = {}
    for group_name, tasks in eval_groups.items():
        group = {}
        for task in tasks:
            if task in raw_task_metrics:
                group[_clean_task_name(task)] = round(raw_task_metrics[task] * 100, 1)
        grouped[group_name] = group
    return grouped


def parse_results():
    """
    Parse all JSONL result files from ModelEvaluationDownstream.

    Filename pattern:
        OLMo2-{size}b-tk{tokens}T-{pt_optim}-Midtrain-{midtrain_tokens}B-
            {olmo|adamw|sam}[-rho{rho}][-hf[-{quant_bit}bit]][-SFT-lr{sft_lr}]
            -downstream-eval.jsonl

    Returns a list of dicts, one per result file.
    """
    raw_results_dir = os.path.join(RESULTS_DIR, "ModelEvaluationDownstream")
    results = []

    for file in sorted(os.listdir(raw_results_dir)):
        if not file.endswith("-downstream-eval.jsonl"):
            continue

        fpath = os.path.join(raw_results_dir, file)
        fname = file.removesuffix("-downstream-eval.jsonl")

        # --- Parse filename fields ---
        # Midtrain tokens (e.g. 50B → 50, 5B → 5)
        midtrain_tokens_match = re.search(r'Midtrain-(\d+)B-', fname)
        midtrain_tokens = int(midtrain_tokens_match.group(1)) if midtrain_tokens_match else None

        # Batch size (e.g. bs1024 → 1024, bs512 → 512)
        bs_match = re.search(r'-bs(\d+)', fname)
        batch_size = int(bs_match.group(1)) if bs_match else None

        # Midtrain optimizer / model tag after "<digits>B-"
        optim_match = re.search(r'Midtrain-\d+B-(sam|adamw|olmo)', fname)
        optimizer = optim_match.group(1) if optim_match else "unknown"

        # SAM rho (only when optimizer is sam)
        rho_val = None
        if optimizer == "sam":
            rho_match = re.search(r'-rho(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', fname)
            if rho_match:
                try:
                    rho_val = float(rho_match.group(1))
                except ValueError:
                    raise ValueError(
                        f"Could not convert rho to float: {rho_match.group(1)!r} in {file}"
                    )

        # SAM anneal flag (e.g. sam-rho5e-2-anneal-bs1024)
        anneal_sam = "-anneal-" in fname

        # Quantization bits (4bit / 8bit)
        quant_bit = None
        quant_match = re.search(r'-(\d+)bit', fname)
        if quant_match:
            quant_bit = int(quant_match.group(1))

        # SFT flag and learning rate
        is_sft = "SFT" in fname
        sft_lr = None
        if is_sft:
            sft_lr_match = re.search(r'SFT-lr([0-9eE\+\-\.]+)', fname)
            if sft_lr_match:
                try:
                    sft_lr = float(sft_lr_match.group(1))
                except ValueError:
                    raise ValueError(
                        f"Could not convert SFT lr to float: "
                        f"{sft_lr_match.group(1)!r} in {file}"
                    )

        # Determine run type
        if is_sft:
            run_type = "sft"
        elif quant_bit is not None:
            run_type = "quant"
        else:
            run_type = "midtrain"

        # Choose correct eval-group definition based on SFT vs base
        eval_groups = SFT_EVAL_GROUPS if is_sft else BASE_EVAL_GROUPS

        # --- Parse JSONL task results ---
        raw_task_metrics = {}
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                task_name = entry.get("task_name", "")
                primary_score = entry.get("metrics", {}).get("primary_score", None)
                if task_name and primary_score is not None:
                    raw_task_metrics[task_name] = primary_score

        # Group into headline eval categories
        task_metrics = _group_task_metrics(raw_task_metrics, eval_groups)

        run_info = {
            "filename": file,
            "optimizer": optimizer,
            "run_type": run_type,
            "task_metrics": task_metrics,
        }
        if midtrain_tokens is not None:
            run_info["midtrain_tokens"] = midtrain_tokens
        if batch_size is not None:
            run_info["batch_size"] = batch_size
        run_info["anneal_sam"] = anneal_sam
        if rho_val is not None:
            run_info["rho"] = rho_val
        if quant_bit is not None:
            run_info["quant_bit"] = quant_bit
        if is_sft:
            run_info["sft_lr"] = sft_lr

        results.append(run_info)

    return results


def get_midtrain_run_info(
    results,
    optim,
    rho=5e-2,
    run_type="midtrain",
    quant_bit=None,
    sft_lr=None,
    midtrain_tokens=None,
    batch_size=None,
    anneal_sam=False,
):
    """
    Filter parsed midtrain results by optimizer, rho, run type, and more.

    Args:
        results: list of dicts returned by parse_results().
        optim: optimizer / tag to filter on ("adamw", "sam", or "olmo").
        rho: SAM rho value to filter on (only used when optim="sam").
        run_type: one of "midtrain", "quant", "sft".
        quant_bit: if run_type="quant", filter to this bit width (4 or 8).
        sft_lr: if run_type is SFT-related, optionally filter to a specific lr.
        midtrain_tokens: filter by midtrain token budget (e.g. 50 or 5).
        batch_size: filter by batch size (e.g. 1024 or 512).
        anneal_sam: if True, only return SAM anneal runs; if False, exclude them.

    Returns:
        A filtered list of run_info dicts, or None if no matches.
    """
    filtered = [r for r in results if r.get("optimizer") == optim]

    # Filter by rho for SAM
    if optim == "sam":
        filtered = [r for r in filtered if r.get("rho") == rho]

    # Filter by anneal_sam
    filtered = [r for r in filtered if r.get("anneal_sam", False) == anneal_sam]

    # Filter by run type
    filtered = [r for r in filtered if r.get("run_type") == run_type]

    # Additional filters per run type
    if run_type == "quant" and quant_bit is not None:
        filtered = [r for r in filtered if r.get("quant_bit") == quant_bit]

    if run_type == "sft" and sft_lr is not None:
        filtered = [r for r in filtered if r.get("sft_lr") == sft_lr]

    if midtrain_tokens is not None:
        filtered = [r for r in filtered if r.get("midtrain_tokens") == midtrain_tokens]

    if batch_size is not None:
        filtered = [r for r in filtered if r.get("batch_size") == batch_size]

    return filtered if filtered else None


def main():
    results = parse_results()

    out_path = os.path.join(RESULTS_DIR, "midtrain_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
