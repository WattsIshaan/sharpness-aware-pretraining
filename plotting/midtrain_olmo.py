"""
Plot OLMo midtrain downstream eval results.

1x3 subplots: one column per eval type (downstream, hf-downstream, hf-quant-4bit-downstream).
X-axis: rho values. Lines for sam, sam-anneal, 4-sam. AdamW as baseline.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from utils.config_globals import RESULTS_DIR
from utils.plotting_globals import (
    BASE_EVAL_TASK_MAP,
    COLOR_MAP,
    FONTSIZE,
    GRID_ALPHA,
    ALPHA,
)
from preprocess_midtrain_olmo import (
    parse_results,
    get_olmo_run_info,
    OLMO_TASK_NAMES,
)

# Rho values present in the OLMo eval files
RHOS = [5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 1e0]
ANNEAL_RHOS = [5e-2, 1e-1, 2e-1]
MSAM_RHOS = [5e-2, 1e-1, 2e-1]

EVAL_TYPES = ["olmo", "hf", "hf-4bit"]

# Display names for OLMo tasks (BASE_EVAL_TASK_MAP may not have all)
OLMO_TASK_DISPLAY = {
    "winogrande": "Winogrande",
    "mmlu": "MMLU",
    "sciq": "SciQ",
    "hellaswag": "HellaSwag",
    "copa": "COPA",
    "openbook_qa": "OpenBookQA",
}

def _rho_label(rho):
    """Format rho as compact string."""
    coeff, exp = f"{rho:.2e}".replace("+", "").split("e")
    coeff = coeff.rstrip("0").rstrip(".")
    return f"{coeff}e{exp}"


def _flatten_task_metrics(run):
    """Return task_metrics dict from run_info."""
    return run.get("task_metrics", {})


def _avg_score(results, eval_type, **kwargs):
    """Average across all 6 tasks for a config."""
    matched = get_olmo_run_info(results, eval_type=eval_type, **kwargs)
    if not matched:
        return None
    flat = _flatten_task_metrics(matched[0])
    vals = [flat[t] for t in OLMO_TASK_NAMES if t in flat and flat[t] is not None]
    return np.mean(vals) if vals else None


def _task_score(results, eval_type, task, **kwargs):
    """Score for a single task."""
    matched = get_olmo_run_info(results, eval_type=eval_type, **kwargs)
    if not matched:
        return None
    flat = _flatten_task_metrics(matched[0])
    return flat.get(task)


def _plot_rho_lines(ax, sam_vals, anneal_vals, msam_vals, adamw_val, ylabel, title):
    """Plot SAM, SAM Anneal, 4-SAM lines + AdamW baseline."""
    rho_labels = [_rho_label(r) for r in RHOS]
    anneal_labels = [_rho_label(r) for r in ANNEAL_RHOS]
    msam_labels = [_rho_label(r) for r in MSAM_RHOS]

    valid = [(l, v) for l, v in zip(rho_labels, sam_vals) if v is not None]
    if valid:
        xl, yl = zip(*valid)
        ax.plot(xl, yl, marker="o", color=COLOR_MAP["sam"], label="SAM", alpha=ALPHA, linewidth=2)

    valid_a = [(l, v) for l, v in zip(anneal_labels, anneal_vals) if v is not None]
    if valid_a:
        xl_a, yl_a = zip(*valid_a)
        ax.plot(
            xl_a, yl_a,
            marker="s",
            color="forestgreen",
            label="SAM Anneal",
            alpha=ALPHA,
            linewidth=2,
            linestyle="--",
        )

    valid_m = [(l, v) for l, v in zip(msam_labels, msam_vals) if v is not None]
    if valid_m:
        xl_m, yl_m = zip(*valid_m)
        ax.plot(
            xl_m, yl_m,
            marker="^",
            color="darkorchid",
            label="4-SAM",
            alpha=ALPHA,
            linewidth=2,
        )

    if adamw_val is not None:
        ax.axhline(
            adamw_val,
            color=COLOR_MAP["adamw"],
            linestyle=":",
            linewidth=2,
            label="AdamW",
        )

    ax.set_xlabel("SAM rho", fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(ylabel, fontsize=FONTSIZE["AXIS"])
    ax.set_title(title, fontsize=FONTSIZE["TITLE"])
    ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=GRID_ALPHA)


def _eval_type_display_name(eval_type):
    """Human-readable eval type label."""
    return {"olmo": "OLMo", "hf": "HF", "hf-4bit": "HF 4-bit"}.get(eval_type, eval_type)


def build_olmo_plots(results):
    """Generate 1x3 plots: average, then one per task."""

    out_dir = os.path.join(RESULTS_DIR, "plots/midtrain_olmo")
    os.makedirs(out_dir, exist_ok=True)

    base_kw = dict(midtrain_tokens=5, batch_size=1024)

    # ── Figure 1: Average across all 6 tasks (1x3) ──
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), sharey="row")

    for col_idx, eval_type in enumerate(EVAL_TYPES):
        ax = axes1[col_idx]
        adamw_avg = _avg_score(results, eval_type=eval_type, optim="adamw", **base_kw)
        sam_avgs = [
            _avg_score(results, eval_type=eval_type, optim="sam", rho=r, **base_kw)
            for r in RHOS
        ]
        anneal_avgs = [
            _avg_score(results, eval_type=eval_type, optim="sam", rho=r, anneal_sam=True, **base_kw)
            for r in ANNEAL_RHOS
        ]
        msam_avgs = [
            _avg_score(results, eval_type=eval_type, optim="sam", rho=r, per_microbatch=True, **base_kw)
            for r in MSAM_RHOS
        ]
        _plot_rho_lines(
            ax, sam_avgs, anneal_avgs, msam_avgs, adamw_avg,
            "Avg Score", _eval_type_display_name(eval_type),
        )

    handles, labels = axes1[-1].get_legend_handles_labels()
    fig1.legend(handles, labels, fontsize=FONTSIZE["LEGEND"] - 1,
                loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4)
    fig1.suptitle("OLMo Midtrain: Average Score vs SAM rho",
                  fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "olmo_avg.pdf"), bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved average plot to {out_dir}/olmo_avg.pdf")

    # ── Figures 2–7: One 1x3 plot per task ──
    for task in OLMO_TASK_NAMES:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey="row")
        task_display = OLMO_TASK_DISPLAY.get(task, BASE_EVAL_TASK_MAP.get(task, task))

        for col_idx, eval_type in enumerate(EVAL_TYPES):
            ax = axes[col_idx]
            adamw_val = _task_score(results, eval_type=eval_type, task=task, optim="adamw", **base_kw)
            sam_vals = [
                _task_score(results, eval_type=eval_type, task=task, optim="sam", rho=r, **base_kw)
                for r in RHOS
            ]
            anneal_vals = [
                _task_score(results, eval_type=eval_type, task=task, optim="sam", rho=r, anneal_sam=True, **base_kw)
                for r in ANNEAL_RHOS
            ]
            msam_vals = [
                _task_score(results, eval_type=eval_type, task=task, optim="sam", rho=r, per_microbatch=True, **base_kw)
                for r in MSAM_RHOS
            ]
            _plot_rho_lines(
                ax, sam_vals, anneal_vals, msam_vals, adamw_val,
                f"{task_display} Score", _eval_type_display_name(eval_type),
            )

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"] - 1,
                   loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4)
        fig.suptitle(f"OLMo Midtrain: {task_display} vs SAM rho",
                    fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
        fig.tight_layout()
        safe_task = task.replace("/", "_")
        fig.savefig(os.path.join(out_dir, f"olmo_{safe_task}.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {task_display} plot to {out_dir}/olmo_{safe_task}.pdf")


def main():
    results = parse_results()
    if not results:
        print("No OLMo results found. Run preprocess_midtrain_olmo.py first.")
        return
    build_olmo_plots(results)


if __name__ == "__main__":
    main()
