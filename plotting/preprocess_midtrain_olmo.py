"""
Parse OLMo eval results:

1) Downstream evals (for previous plots): ModelEvaluationDownstreamOLMo only.
   Files with -downstream-eval.json, -hf-downstream-eval.json, -hf-quant-4bit-downstream-eval.json.
   Excludes "CPT" in filename. run_type="downstream". Use get_olmo_run_info().

2) CPT runs (for Pareto): ModelEvaluation + ModelEvaluationDownstreamOLMo.
   Pairs with "CPT" in name, matched by base name. run_type="cpt". Use get_cpt_run_info().
"""

import json
import os
import re
from utils.config_globals import RESULTS_DIR


# ---------------------------------------------------------------------------
# OLMo downstream eval tasks (base evals for Pareto x-axis)
# ---------------------------------------------------------------------------

OLMO_DOWNSTREAM_KEYS = [
    "eval/downstream/winogrande_acc",
    "eval/downstream/mmlu_other_var_len_norm",
    "eval/downstream/sciq_acc",
    "eval/downstream/hellaswag_len_norm",
    "eval/downstream/copa_acc",
    "eval/downstream/openbook_qa_len_norm",
]

OLMO_TASK_NAMES = [
    "winogrande",
    "mmlu",
    "sciq",
    "hellaswag",
    "copa",
    "openbook_qa",
]

OLMO_KEY_TO_TASK = dict(zip(OLMO_DOWNSTREAM_KEYS, OLMO_TASK_NAMES))

# Eval type suffixes for non-CPT downstream files (kept for compatibility)
EVAL_TYPE_SUFFIXES = [
    "-hf-quant-4bit-downstream-eval.json",
    "-hf-downstream-eval.json",
    "-downstream-eval.json",
]
EVAL_TYPE_NAMES = {
    "-hf-quant-4bit-downstream-eval.json": "hf-4bit",
    "-hf-downstream-eval.json": "hf",
    "-downstream-eval.json": "olmo",
}

# CPT file suffixes
CPT_EVAL_SUFFIX = "-eval.json"
CPT_DOWNSTREAM_SUFFIX = "-downstream-eval.json"
# ModelEvaluationDownstream uses JSONL with -hf-downstream-eval.jsonl
CPT_DOWNSTREAM_JSONL_SUFFIX = "-hf-downstream-eval.jsonl"
# Non-CPT OLMES downstream files: *-downstream-eval.jsonl (base name has -hf or -hf-4bit)
DOWNSTREAM_OLMES_JSONL_SUFFIX = "-downstream-eval.jsonl"

# ---------------------------------------------------------------------------
# OLMES eval groups (base model; keys as in JSONL, clean names for lookups)
# ---------------------------------------------------------------------------

BASE_MCQ_TASKS = [
    "arc_challenge::olmes",
    "boolq::olmes",
    "hellaswag::olmes",
    "mmlu:mc::olmes",
    "winogrande::olmes",
]
BASE_GENERATIVE_TASKS = ["drop", "naturalqs_open"]
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

# Tasks that appear both as plain (e.g. hellaswag::olmes) and :mc / :rc rows in JSONL — keep separate keys.
_OLMES_MC_RC_DISAMBIG = frozenset({"hellaswag", "winogrande", "arc_challenge"})


def _clean_olmes_task_name(name: str) -> str:
    """
    Normalize OLMES JSONL task_name -> dict key (scores 0–100).

    Strips trailing ``::olmes`` / ``::tulu`` / etc. For most tasks, keeps only the
    segment before the first ``:`` (e.g. ``mmlu:mc::olmes`` -> ``mmlu``,
    ``agi_eval_english:1shot::olmes`` -> ``agi_eval_english``).

    For ``hellaswag``, ``winogrande``, and ``arc_challenge``, preserves ``:mc`` and
    ``:rc`` so MC vs RC vs aggregate (no suffix) do not overwrite each other.
    """
    if "::" in name:
        name = name.split("::")[0]
    parts = name.split(":")
    if len(parts) >= 2:
        base, suf = parts[0], parts[1]
        if base in _OLMES_MC_RC_DISAMBIG and suf in ("mc", "rc"):
            return f"{base}:{suf}"
    return parts[0]


OLMES_EVAL_GROUPS = {
    group: [_clean_olmes_task_name(t) for t in tasks]
    for group, tasks in BASE_EVAL_GROUPS.items()
}
OLMES_EVAL_TASKS = [t for tasks in OLMES_EVAL_GROUPS.values() for t in tasks]

# Key in ModelEvaluation ``*-eval.json`` for CPT finetuning val loss: ``input_ids-{suffix}``.
# Suffix usually matches ``cpt_dataset``; Meta-Math logs ``metamath`` (val ``input_ids-metamath.npy``),
# not ``meta-math``.
_CPT_FINETUNE_LOSS_SUFFIX = {
    "meta-math": "metamath",
}


def _cpt_finetuning_val_loss_key(cpt_dataset: str) -> str:
    suffix = _CPT_FINETUNE_LOSS_SUFFIX.get(cpt_dataset, cpt_dataset)
    return f"input_ids-{suffix}"


def _parse_task_metrics(raw_data: dict) -> dict:
    """Extract task_name -> score (0-100) from raw downstream JSON."""
    out = {}
    for key, val in raw_data.items():
        if key in OLMO_KEY_TO_TASK and val is not None:
            task = OLMO_KEY_TO_TASK[key]
            out[task] = round(float(val) * 100, 1)
    return out


def _parse_task_metrics_from_jsonl(fpath: str) -> dict:
    """
    Parse task_name -> score (0-100) from ModelEvaluationDownstream JSONL.
    Each line: {"task_name": "...", "metrics": {"primary_score": 0.xx}}.
    """
    out = {}
    with open(fpath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_name = entry.get("task_name", "")
            primary_score = entry.get("metrics", {}).get("primary_score")
            if task_name and primary_score is not None:
                # Use clean task name to match OLMES_EVAL_GROUPS (e.g. "mmlu:mc::olmes" -> "mmlu", "gsm8k::olmes" -> "gsm8k")
                clean_name = _clean_olmes_task_name(task_name.split("::")[0] if "::" in task_name else task_name)
                out[clean_name] = round(float(primary_score) * 100, 1)
    return out


def _parse_downstream_olmes_lookup():
    """
    Parse ModelEvaluationDownstream JSONL files that are not CPT and not SFT.
    Returns dict: base_name (no suffix) -> olmes_eval (task -> score 0-100).
    Used to attach OLMES evals to downstream runs from ModelEvaluationDownstreamOLMo.
    """
    med_dir = os.path.join(RESULTS_DIR, "ModelEvaluationDownstream")
    if not os.path.isdir(med_dir):
        return {}
    lookup = {}
    for file in os.listdir(med_dir):
        if "CPT" in file or "SFT" in file or not file.endswith(DOWNSTREAM_OLMES_JSONL_SUFFIX):
            continue
        base = file.removesuffix(DOWNSTREAM_OLMES_JSONL_SUFFIX)
        fpath = os.path.join(med_dir, file)
        olmes_eval = _parse_task_metrics_from_jsonl(fpath)
        if olmes_eval:
            lookup[base] = olmes_eval
    return lookup


def _downstream_run_to_olmes_key(filename: str, eval_type: str):
    """
    From a ModelEvaluationDownstreamOLMo filename and eval_type, get the key to look up olmes_eval.
    OLMo suffix is e.g. -hf-downstream-eval.json (so fname loses -hf); OLMES base is ...-hf.
    For hf-4bit we remove -hf-quant-4bit-downstream-eval.json so fname is base; OLMES base is base-hf-4bit.
    """
    for suffix in EVAL_TYPE_SUFFIXES:
        if filename.endswith(suffix):
            fname = filename.removesuffix(suffix)
            if eval_type == "olmo":
                return None  # no OLMES file for native olmo eval
            if eval_type == "hf":
                return fname + "-hf"
            if eval_type == "hf-4bit":
                # Removing "-hf-quant-4bit-downstream-eval.json" drops the quant tag from fname.
                # OLMES base uses "-hf-4bit" instead.
                return fname if fname.endswith("-hf-4bit") else fname + "-hf-4bit"
            return fname
    return None


def _parse_cpt_basename(fname: str):
    """
    Parse CPT run metadata from base filename (no suffix).
    Example: OLMo2-1b-tk4T-adamw-Midtrain-5B-adamw-bs1024-CPT-musicpile-tk50M-lr1.00e-5-wd0-bs64
    """
    out = {
        "pretrain_token": None,
        "midtrain_tokens": None,
        "optimizer": None,
        "rho": None,
        "batch_size": None,
        "cpt_dataset": None,
        "cpt_tokens": None,
        "cpt_lr": None,
        "cpt_wd": None,
        "cpt_bs": None,
        "step": -1,
    }
    # Pretrain tokens (e.g. tk4T -> 4, tk2T -> 2)
    m = re.search(r"tk(\d+)T", fname)
    if m:
        out["pretrain_token"] = int(m.group(1))
    # Midtrain tokens
    m = re.search(r"Midtrain-(\d+)B-", fname)
    if m:
        out["midtrain_tokens"] = int(m.group(1))
    # Optimizer and optional rho
    m = re.search(r"Midtrain-\d+B-(adamw|sam)(?:-rho((?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?))?-bs(\d+)", fname)
    if m:
        out["optimizer"] = m.group(1)
        if m.group(2):
            try:
                out["rho"] = float(m.group(2))
            except ValueError:
                pass
        if m.group(3):
            out["batch_size"] = int(m.group(3))
    # CPT block: CPT-{dataset}-tk{N}M-lr{lr}-wd{wd}-bs{bs}
    # dataset: musicpile, tulu, alpaca, siqa, gsm8k, codealpaca, stackmathqa (and starcoder for legacy)
    m = re.search(
        r"CPT-(musicpile|starcoder|tulu|tuluv2|magicoder|meta-math|alpaca|siqa|gsm8k|codealpaca|stackmathqa)-tk(\d+)M-lr([\d.eE+-]+)-wd([\d.]+)-bs(\d+)",
        fname,
    )
    if m:
        out["cpt_dataset"] = m.group(1)
        out["cpt_tokens"] = int(m.group(2))  # 50M -> 50
        try:
            out["cpt_lr"] = float(m.group(3))
        except ValueError:
            pass
        try:
            out["cpt_wd"] = float(m.group(4))
        except ValueError:
            pass
        out["cpt_bs"] = int(m.group(5))
    # CPT eval step: step4000 in filename -> 4000, else -1
    step_m = re.search(r"step(\d+)", fname, re.IGNORECASE)
    if step_m:
        out["step"] = int(step_m.group(1))
    return out


def parse_downstream_results():
    """
    Parse non-CPT downstream evals from ModelEvaluationDownstreamOLMo (for previous plots).
    Only files matching EVAL_TYPE_SUFFIXES and without "CPT" in filename.
    """
    downstream_dir = os.path.join(RESULTS_DIR, "ModelEvaluationDownstreamOLMo")
    if not os.path.isdir(downstream_dir):
        return []

    results = []
    for file in sorted(os.listdir(downstream_dir)):
        if "CPT" in file or not file.endswith(".json"):
            continue
        eval_type = None
        for suffix in EVAL_TYPE_SUFFIXES:
            if file.endswith(suffix):
                eval_type = EVAL_TYPE_NAMES[suffix]
                fname = file.removesuffix(suffix)
                break
        if eval_type is None:
            continue

        fpath = os.path.join(downstream_dir, file)
        model_type = "hf" if "-hf-" in file else "olmo"

        pretrain_token_match = re.search(r"tk(\d+)T", fname)
        pretrain_token = int(pretrain_token_match.group(1)) if pretrain_token_match else None
        midtrain_tokens_match = re.search(r"Midtrain-(\d+)B-", fname)
        midtrain_tokens = int(midtrain_tokens_match.group(1)) if midtrain_tokens_match else None
        bs_match = re.search(r"-bs(\d+)", fname)
        batch_size = int(bs_match.group(1)) if bs_match else None
        optim_match = re.search(r"Midtrain-\d+B-(sam_per_microbatch|sam|adamw|olmo)", fname)
        optimizer = optim_match.group(1) if optim_match else "unknown"
        per_microbatch = optimizer == "sam_per_microbatch"
        if per_microbatch:
            optimizer = "sam"
        rho_val = None
        if optimizer == "sam":
            rho_match = re.search(r"-rho((?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)", fname)
            if rho_match:
                try:
                    rho_val = float(rho_match.group(1))
                except ValueError:
                    pass
        anneal_sam = "-anneal-" in fname

        with open(fpath, "r") as f:
            raw_data = json.load(f)
        task_metrics = _parse_task_metrics(raw_data)

        run_info = {
            "filename": file,
            "run_type": "downstream",
            "optimizer": optimizer,
            "eval_type": eval_type,
            "model_type": model_type,
            "task_metrics": task_metrics,
        }
        if pretrain_token is not None:
            run_info["pretrain_token"] = pretrain_token
        if midtrain_tokens is not None:
            run_info["midtrain_tokens"] = midtrain_tokens
        if batch_size is not None:
            run_info["batch_size"] = batch_size
        run_info["anneal_sam"] = anneal_sam
        run_info["per_microbatch"] = per_microbatch
        if rho_val is not None:
            run_info["rho"] = rho_val
        results.append(run_info)

    # Attach OLMES evals from ModelEvaluationDownstream (non-CPT, non-SFT)
    olmes_lookup = _parse_downstream_olmes_lookup()
    for run_info in results:
        key = _downstream_run_to_olmes_key(run_info["filename"], run_info["eval_type"])
        if key and key in olmes_lookup:
            run_info["olmes_eval"] = olmes_lookup[key]

    # Add runs from ModelEvaluationDownstream JSONL that have no OLMo .json (e.g. rho 1.25e-1, 1.5e-1)
    existing_keys = {
        (
            r.get("pretrain_token"),
            r.get("midtrain_tokens"),
            r.get("eval_type"),
            r.get("optimizer"),
            r.get("rho"),
            r.get("anneal_sam", False),
            r.get("per_microbatch", False),
            r.get("batch_size"),
        )
        for r in results
    }
    med_dir = os.path.join(RESULTS_DIR, "ModelEvaluationDownstream")
    if os.path.isdir(med_dir):
        for file in sorted(os.listdir(med_dir)):
            if "CPT" in file or "SFT" in file or not file.endswith(DOWNSTREAM_OLMES_JSONL_SUFFIX):
                continue
            base = file.removesuffix(DOWNSTREAM_OLMES_JSONL_SUFFIX)
            if base.endswith("-hf-4bit"):
                eval_type = "hf-4bit"
                core = base.removesuffix("-hf-4bit")
            elif base.endswith("-hf"):
                eval_type = "hf"
                core = base.removesuffix("-hf")
            else:
                continue
            fpath = os.path.join(med_dir, file)
            olmes_eval = _parse_task_metrics_from_jsonl(fpath)
            if not olmes_eval:
                continue
            pretrain_token = None
            pt_m = re.search(r"tk(\d+)T", core)
            if pt_m:
                pretrain_token = int(pt_m.group(1))
            midtrain_tokens = None
            mt_m = re.search(r"Midtrain-(\d+)B-", core)
            if mt_m:
                midtrain_tokens = int(mt_m.group(1))
            bs_m = re.search(r"-bs(\d+)", core)
            batch_size = int(bs_m.group(1)) if bs_m else None
            optim_m = re.search(r"Midtrain-\d+B-(sam_per_microbatch|sam|adamw)", core)
            optimizer = optim_m.group(1) if optim_m else "unknown"
            per_microbatch = optimizer == "sam_per_microbatch"
            if per_microbatch:
                optimizer = "sam"
            rho_val = None
            if optimizer == "sam":
                rho_m = re.search(r"-rho((?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)", core)
                if rho_m:
                    try:
                        rho_val = float(rho_m.group(1))
                    except ValueError:
                        pass
            anneal_sam = "-anneal-" in core
            key = (pretrain_token, midtrain_tokens, eval_type, optimizer, rho_val, anneal_sam, per_microbatch, batch_size)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            run_info = {
                "filename": file,
                "run_type": "downstream",
                "optimizer": optimizer,
                "eval_type": eval_type,
                "model_type": "hf",
                "task_metrics": {},
                "olmes_eval": olmes_eval,
                "anneal_sam": anneal_sam,
                "per_microbatch": per_microbatch,
            }
            if pretrain_token is not None:
                run_info["pretrain_token"] = pretrain_token
            if midtrain_tokens is not None:
                run_info["midtrain_tokens"] = midtrain_tokens
            if batch_size is not None:
                run_info["batch_size"] = batch_size
            if rho_val is not None:
                run_info["rho"] = rho_val
            results.append(run_info)

    return results


def parse_cpt_results():
    """
    Parse CPT runs from ModelEvaluation + ModelEvaluationDownstreamOLMo (merged pairs).
    Only "CPT" in filename. run_type="cpt".
    """
    eval_dir = os.path.join(RESULTS_DIR, "ModelEvaluation")
    downstream_dir = os.path.join(RESULTS_DIR, "ModelEvaluationDownstreamOLMo")
    if not os.path.isdir(eval_dir) or not os.path.isdir(downstream_dir):
        return []

    eval_files = {}
    for f in os.listdir(eval_dir):
        if not f.endswith(CPT_EVAL_SUFFIX) or "CPT" not in f:
            continue
        base = f.removesuffix(CPT_EVAL_SUFFIX)
        eval_files[base] = os.path.join(eval_dir, f)

    downstream_files = {}
    for f in os.listdir(downstream_dir):
        if not f.endswith(CPT_DOWNSTREAM_SUFFIX) or "CPT" not in f:
            continue
        base = f.removesuffix(CPT_DOWNSTREAM_SUFFIX)
        downstream_files[base] = os.path.join(downstream_dir, f)

    results = []
    for base in sorted(eval_files.keys()):
        if base not in downstream_files:
            continue
        meta = _parse_cpt_basename(base)
        if meta["cpt_dataset"] is None or meta["optimizer"] is None:
            continue

        loss_key = _cpt_finetuning_val_loss_key(meta["cpt_dataset"])
        with open(eval_files[base], "r") as f:
            eval_data = json.load(f)
        finetuning_val_loss = eval_data.get(loss_key)
        if finetuning_val_loss is not None:
            finetuning_val_loss = float(finetuning_val_loss)

        with open(downstream_files[base], "r") as f:
            downstream_data = json.load(f)
        task_metrics = _parse_task_metrics(downstream_data)

        run_info = {
            "filename": base + CPT_EVAL_SUFFIX,
            "run_type": "cpt",
            "finetuning_val_loss": finetuning_val_loss,
            "task_metrics": task_metrics,
            "optimizer": meta["optimizer"],
            "midtrain_tokens": meta["midtrain_tokens"],
            "batch_size": meta["batch_size"],
            "cpt_dataset": meta["cpt_dataset"],
            "cpt_tokens": meta["cpt_tokens"],
            "cpt_lr": meta["cpt_lr"],
            "cpt_wd": meta["cpt_wd"],
            "cpt_bs": meta["cpt_bs"],
            "step": meta.get("step", -1),
        }
        if meta["pretrain_token"] is not None:
            run_info["pretrain_token"] = meta["pretrain_token"]
        if meta["rho"] is not None:
            run_info["rho"] = meta["rho"]
        results.append(run_info)
    return results


def parse_cpt_results_from_model_eval_downstream():
    """
    Parse CPT runs from ModelEvaluation + ModelEvaluationDownstream (JSONL format).
    ModelEvaluationDownstream files: *-hf-downstream-eval.jsonl with "CPT" in name.
    Matches by base name with ModelEvaluation *-eval.json for finetuning_val_loss.
    """
    eval_dir = os.path.join(RESULTS_DIR, "ModelEvaluation")
    downstream_dir = os.path.join(RESULTS_DIR, "ModelEvaluationDownstream")
    if not os.path.isdir(eval_dir) or not os.path.isdir(downstream_dir):
        return []

    eval_files = {}
    for f in os.listdir(eval_dir):
        if not f.endswith(CPT_EVAL_SUFFIX) or "CPT" not in f:
            continue
        base = f.removesuffix(CPT_EVAL_SUFFIX)
        eval_files[base] = os.path.join(eval_dir, f)

    results = []
    for f in sorted(os.listdir(downstream_dir)):
        if "CPT" not in f or not f.endswith(CPT_DOWNSTREAM_JSONL_SUFFIX):
            continue
        base = f.removesuffix(CPT_DOWNSTREAM_JSONL_SUFFIX)
        meta = _parse_cpt_basename(base)
        if meta["cpt_dataset"] is None or meta["optimizer"] is None:
            continue

        finetuning_val_loss = None
        if base in eval_files:
            loss_key = _cpt_finetuning_val_loss_key(meta["cpt_dataset"])
            with open(eval_files[base], "r") as ef:
                eval_data = json.load(ef)
            finetuning_val_loss = eval_data.get(loss_key)
            if finetuning_val_loss is not None:
                finetuning_val_loss = float(finetuning_val_loss)

        downstream_path = os.path.join(downstream_dir, f)
        olmes_eval = _parse_task_metrics_from_jsonl(downstream_path)
        if not olmes_eval:
            continue

        run_info = {
            "filename": base + CPT_EVAL_SUFFIX,
            "run_type": "cpt",
            "finetuning_val_loss": finetuning_val_loss,
            "task_metrics": {},
            "olmes_eval": olmes_eval,
            "optimizer": meta["optimizer"],
            "midtrain_tokens": meta["midtrain_tokens"],
            "batch_size": meta["batch_size"],
            "cpt_dataset": meta["cpt_dataset"],
            "cpt_tokens": meta["cpt_tokens"],
            "cpt_lr": meta["cpt_lr"],
            "cpt_wd": meta["cpt_wd"],
            "cpt_bs": meta["cpt_bs"],
            "step": meta.get("step", -1),
            "source": "ModelEvaluationDownstream",
        }
        if meta["pretrain_token"] is not None:
            run_info["pretrain_token"] = meta["pretrain_token"]
        if meta["rho"] is not None:
            run_info["rho"] = meta["rho"]
        results.append(run_info)
    return results


def parse_sft_results():
    """
    Parse SFT runs from ModelEvaluationDownstream (JSONL).
    Only files with "SFT" in name and without "CPT". run_type="sft".
    Each entry has olmes_eval only (no finetuning_val_loss or task_metrics).
    """
    downstream_dir = os.path.join(RESULTS_DIR, "ModelEvaluationDownstream")
    if not os.path.isdir(downstream_dir):
        return []

    results = []
    for file in sorted(os.listdir(downstream_dir)):
        if "CPT" in file or "SFT" not in file or not file.endswith("-downstream-eval.jsonl"):
            continue

        fpath = os.path.join(downstream_dir, file)
        fname = file.removesuffix("-downstream-eval.jsonl")

        pretrain_token_match = re.search(r"tk(\d+)T", fname)
        pretrain_token = int(pretrain_token_match.group(1)) if pretrain_token_match else None
        midtrain_tokens_match = re.search(r"Midtrain-(\d+)B-", fname)
        midtrain_tokens = int(midtrain_tokens_match.group(1)) if midtrain_tokens_match else None
        optim_match = re.search(r"Midtrain-\d+B-(sam_per_microbatch|sam|adamw)", fname)
        optimizer = optim_match.group(1) if optim_match else "unknown"
        if optimizer == "sam_per_microbatch":
            optimizer = "sam"
        rho_val = None
        if optimizer == "sam":
            rho_match = re.search(r"-rho((?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)", fname)
            if rho_match:
                try:
                    rho_val = float(rho_match.group(1))
                except ValueError:
                    pass
        sft_lr = None
        sft_lr_match = re.search(r"SFT-lr([\d.eE+-]+)", fname)
        if sft_lr_match:
            try:
                sft_lr = float(sft_lr_match.group(1))
            except ValueError:
                pass

        olmes_eval = _parse_task_metrics_from_jsonl(fpath)
        if not olmes_eval:
            continue

        run_info = {
            "filename": file,
            "run_type": "sft",
            "olmes_eval": olmes_eval,
            "optimizer": optimizer,
            "sft_lr": sft_lr,
        }
        if pretrain_token is not None:
            run_info["pretrain_token"] = pretrain_token
        if midtrain_tokens is not None:
            run_info["midtrain_tokens"] = midtrain_tokens
        if rho_val is not None:
            run_info["rho"] = rho_val
        results.append(run_info)
    return results


def parse_results():
    """
    Parse all: downstream evals (for previous plots) + CPT merged runs (for Pareto).
    CPT from ModelEvaluation+ModelEvaluationDownstreamOLMo and ModelEvaluation+ModelEvaluationDownstream.
    Merge task_metrics: ModelEvaluationDownstream JSONL has extra eval tasks (e.g. gsm8k) that we add
    to runs from ModelEvaluationDownstreamOLMo.
    """
    downstream = parse_downstream_results()
    cpt_olmo = parse_cpt_results()
    cpt_downstream = parse_cpt_results_from_model_eval_downstream()

    # Build lookup: key -> run for cpt_olmo (include pretrain_token and step so 2T/4T and step4000 don't merge)
    cpt_by_key = {}
    for r in cpt_olmo:
        key = (r.get("pretrain_token"), r.get("midtrain_tokens"), r.get("cpt_dataset"), r.get("cpt_tokens"),
               r.get("optimizer"), r.get("rho"), r.get("cpt_lr"), r.get("step", -1))
        cpt_by_key[key] = r

    # Merge: add olmes_eval (gsm8k, etc.) from ModelEvaluationDownstream JSONL into existing runs
    for r in cpt_downstream:
        key = (r.get("pretrain_token"), r.get("midtrain_tokens"), r.get("cpt_dataset"), r.get("cpt_tokens"),
               r.get("optimizer"), r.get("rho"), r.get("cpt_lr"), r.get("step", -1))
        if key in cpt_by_key:
            # Add olmes_eval from JSONL (gsm8k and other OLMES eval tasks)
            existing = cpt_by_key[key].setdefault("olmes_eval", {})
            for task, score in (r.get("olmes_eval") or {}).items():
                existing[task] = score
        else:
            cpt_by_key[key] = r
            cpt_olmo.append(r)

    sft = parse_sft_results()
    return downstream + cpt_olmo + sft


def filter_results(
    results,
    optimizer=None,
    pretrain_token=None,
    midtrain_token=None,
    cpt_dataset=None,
    cpt_tokens=None,
    finetune_type=None,
    rho=None,
    sft_lr=None,
    eval_type=None,
    anneal_sam=False,
    per_microbatch=False,
    batch_size=None,
):
    """
    Single entry point to filter results by common criteria.

    finetune_type: "CPT" | "SFT" | None. If "CPT", only run_type=="cpt" and
        cpt_dataset/cpt_tokens apply. If "SFT", only run_type=="sft".
    midtrain_token: midtrain tokens in B (e.g. 5 or 50).
    pretrain_token: pretrain tokens in trillions (e.g. 4 for 4T).
    """
    filtered = list(results)
    if finetune_type == "CPT":
        filtered = [r for r in filtered if r.get("run_type") == "cpt"]
    elif finetune_type == "SFT":
        filtered = [r for r in filtered if r.get("run_type") == "sft"]

    if optimizer is not None:
        filtered = [r for r in filtered if r.get("optimizer") == optimizer]
    if pretrain_token is not None:
        filtered = [r for r in filtered if r.get("pretrain_token") == pretrain_token]
    if midtrain_token is not None:
        filtered = [r for r in filtered if r.get("midtrain_tokens") == midtrain_token]
    if rho is not None:
        filtered = [r for r in filtered if r.get("rho") == rho]
    if sft_lr is not None:
        filtered = [r for r in filtered if r.get("sft_lr") == sft_lr]
    if eval_type is not None:
        filtered = [r for r in filtered if r.get("eval_type") == eval_type]
    if batch_size is not None:
        filtered = [r for r in filtered if r.get("batch_size") == batch_size]
    filtered = [r for r in filtered if r.get("anneal_sam", False) == anneal_sam]
    filtered = [r for r in filtered if r.get("per_microbatch", False) == per_microbatch]

    if finetune_type == "CPT":
        if cpt_dataset is not None:
            filtered = [r for r in filtered if r.get("cpt_dataset") == cpt_dataset]
        if cpt_tokens is not None:
            filtered = [r for r in filtered if r.get("cpt_tokens") == cpt_tokens]

    return filtered


def get_olmo_run_info(
    results,
    optim,
    rho=None,
    eval_type=None,
    midtrain_tokens=None,
    batch_size=None,
    anneal_sam=False,
    per_microbatch=False,
    pretrain_token=None,
):
    """
    Filter parsed results for downstream evals (run_type="downstream").
    Used by midtrain_olmo.py and midtrain_1b.py.

    Returns:
        List of matching run_info dicts, or None if no matches.
    """
    filtered = [r for r in results if r.get("run_type") == "downstream"]
    filtered = [r for r in filtered if r.get("optimizer") == optim]

    if optim == "sam" and rho is not None:
        filtered = [r for r in filtered if r.get("rho") == rho]
    filtered = [r for r in filtered if r.get("anneal_sam", False) == anneal_sam]
    filtered = [r for r in filtered if r.get("per_microbatch", False) == per_microbatch]
    if eval_type is not None:
        filtered = [r for r in filtered if r.get("eval_type") == eval_type]
    if midtrain_tokens is not None:
        filtered = [r for r in filtered if r.get("midtrain_tokens") == midtrain_tokens]
    if batch_size is not None:
        filtered = [r for r in filtered if r.get("batch_size") == batch_size]
    if pretrain_token is not None:
        filtered = [r for r in filtered if r.get("pretrain_token") == pretrain_token]

    return filtered if filtered else None


def get_cpt_run_info(
    results,
    midtrain_tokens=None,
    cpt_dataset=None,
    cpt_tokens=None,
    optim=None,
    rho=None,
    cpt_lr=None,
    cpt_wd=None,
    cpt_bs=None,
    pretrain_token=None,
    step=None,
):
    """
    Filter parsed CPT results by hyperparameters (like helper.get_run_info).

    pretrain_token: filter by pretrain token count in trillions (e.g. 2 for 2T, 4 for 4T).
    step: eval step from filename (e.g. 4000 for step4000); default -1 when not in filename.

    Returns:
        List of matching run_info dicts, or None if no matches.
    """
    filtered = [r for r in results if r.get("run_type") == "cpt"]

    if pretrain_token is not None:
        filtered = [r for r in filtered if r.get("pretrain_token") == pretrain_token]
    if midtrain_tokens is not None:
        filtered = [r for r in filtered if r.get("midtrain_tokens") == midtrain_tokens]
    if cpt_dataset is not None:
        filtered = [r for r in filtered if r.get("cpt_dataset") == cpt_dataset]
    if cpt_tokens is not None:
        filtered = [r for r in filtered if r.get("cpt_tokens") == cpt_tokens]
    if optim is not None:
        filtered = [r for r in filtered if r.get("optimizer") == optim]
    if optim == "sam" and rho is not None:
        filtered = [r for r in filtered if r.get("rho") == rho]
    if cpt_lr is not None:
        filtered = [r for r in filtered if r.get("cpt_lr") == cpt_lr]
    if cpt_wd is not None:
        filtered = [r for r in filtered if r.get("cpt_wd") == cpt_wd]
    if cpt_bs is not None:
        filtered = [r for r in filtered if r.get("cpt_bs") == cpt_bs]
    if step is not None:
        filtered = [r for r in filtered if r.get("step", -1) == step]

    return filtered if filtered else None


def main():
    results = parse_results()
    out_path = os.path.join(RESULTS_DIR, "midtrain_olmo_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    cpt_count = sum(1 for r in results if r.get("run_type") == "cpt")
    sft_count = sum(1 for r in results if r.get("run_type") == "sft")
    downstream_count = len(results) - cpt_count - sft_count
    print(f"Saved {downstream_count} downstream + {cpt_count} CPT + {sft_count} SFT = {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
