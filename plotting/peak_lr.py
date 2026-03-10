import matplotlib.pyplot as plt
import numpy as np
import os
import json

from utils.config_globals import RESULTS_DIR, PERTURBATIONS, TOKEN_LIST
from utils.plotting_globals import (
    FONTSIZE, FIG_WIDTH, FIG_HEIGHT, COLOR_MAP,
    LEGEND_PARAM, ALPHA, GRID_ALPHA, MARKERS, YLABEL,
)
from utils.helper import get_run_info

PEAK_LRS = [1e-4, 3e-4, 6e-4]
LR_LABELS = {1e-4: "1e-4", 3e-4: "3e-4", 6e-4: "6e-4"}
LR_COLORS = {1e-4: COLOR_MAP[1e-4], 3e-4: COLOR_MAP[3e-4], 6e-4: COLOR_MAP[6e-4]}
SIZE = 60
TOKEN = TOKEN_LIST[SIZE][-1]  # 192


# ── 1.1  Quantisation bar plot ──────────────────────────────────────────────

def plot_quant_bar(results):
    """Bar plot: 2 groups (Cosine, WSD) × 3 bars (peak LRs) showing 4-bit dclm_val."""

    groups = ["Cosine", "WSD"]
    n_groups = len(groups)
    n_bars = len(PEAK_LRS)
    bar_width = 0.18
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.6, FIG_HEIGHT * 1.2))

    all_vals = []
    for i, lr in enumerate(PEAK_LRS):
        vals = []

        # Cosine
        run_info = get_run_info(results, SIZE, "adamw", pt_lr=lr, quantized=True, model_type="hf")
        if run_info and run_info["quantized"].get(4, {}).get(TOKEN):
            vals.append(run_info["quantized"][4][TOKEN]["dclm_quant"])
        else:
            vals.append(None)

        # WSD
        run_info = get_run_info(
            results, SIZE, "adamw", pt_lr=lr, quantized=True, model_type="hf",
            anneal=True, anneal_percent=10, anneal_optim="adamw",
        )
        if run_info and run_info["quantized"].get(4, {}).get(TOKEN):
            vals.append(run_info["quantized"][4][TOKEN]["dclm_quant"])
        else:
            vals.append(None)

        all_vals.extend([v for v in vals if v is not None])

        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(
            x + offset, vals, bar_width,
            label=LR_LABELS[lr],
            color=LR_COLORS[lr],
            alpha=ALPHA,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(YLABEL["PT_LOSS_QUANTIZED"], fontsize=FONTSIZE["AXIS"])
    ax.tick_params(axis="y", labelsize=FONTSIZE["TICKS"])
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_title("4-bit Quantisation", fontsize=FONTSIZE["TITLE"])

    # ylim ±5% of min/max
    if all_vals:
        y_min, y_max = min(all_vals), max(all_vals)
        ax.set_ylim(y_min * 0.95, y_max * 1.05)

    # Legend at bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Peak LR",
        fontsize=FONTSIZE["LEGEND"],
        title_fontsize=FONTSIZE["LEGEND"],
        loc=LEGEND_PARAM["LOC"],
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(handles),
    )

    plt.tight_layout()
    out_dir = os.path.join(RESULTS_DIR, "plots/peak_lr")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "quant_bar.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved quant_bar.pdf")


# ── 1.2  Perturbation line plot (1×2 subplots: Cosine | WSD) ────────────────

def plot_perturbation(results):
    """Line plot: dclm_val (y) vs gamma (x), 3 lines per LR, 1×2 subplot for Cosine / WSD."""

    perturbations = sorted(PERTURBATIONS)

    schedules = [
        ("Cosine", dict(anneal=False)),
        ("WSD",    dict(anneal=True, anneal_percent=10, anneal_optim="adamw")),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.3, FIG_HEIGHT * 1.2), sharey=True)

    for col, (title, extra_kw) in enumerate(schedules):
        ax = axs[col]
        for i, lr in enumerate(PEAK_LRS):
            run_info = get_run_info(results, SIZE, "adamw", pt_lr=lr, perturb=True, **extra_kw)
            if run_info is None:
                continue

            x_vals, y_vals = [], []
            for gamma in perturbations:
                entry = run_info["perturbed"].get(gamma, {}).get(TOKEN)
                if entry:
                    x_vals.append(gamma)
                    y_vals.append(entry["dclm_perturbed"])

            if x_vals:
                ax.plot(
                    x_vals, y_vals,
                    label=LR_LABELS[lr],
                    marker=MARKERS[i],
                    color=LR_COLORS[lr],
                    alpha=ALPHA,
                )

        ax.set_xlabel("Perturbation (γ)", fontsize=FONTSIZE["AXIS"])
        if col == 0:
            ax.set_ylabel(YLABEL["PT_LOSS_PERTURBED"], fontsize=FONTSIZE["AXIS"])
        ax.set_title(title, fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"])
        ax.grid(True, alpha=GRID_ALPHA)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Peak LR",
        fontsize=FONTSIZE["LEGEND"],
        title_fontsize=FONTSIZE["LEGEND"],
        loc=LEGEND_PARAM["LOC"],
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(handles),
    )
    plt.tight_layout()
    out_dir = os.path.join(RESULTS_DIR, "plots/peak_lr")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "perturbation.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved perturbation.pdf")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as f:
        results = json.load(f)

    out_dir = os.path.join(RESULTS_DIR, "plots/peak_lr")
    os.makedirs(out_dir, exist_ok=True)

    plot_quant_bar(results)
    plot_perturbation(results)


if __name__ == "__main__":
    main()
