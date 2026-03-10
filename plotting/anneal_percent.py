import matplotlib.pyplot as plt
import numpy as np
import os
import json

from utils.config_globals import RESULTS_DIR, PERTURBATIONS, TOKEN_LIST
from utils.plotting_globals import (
    FONTSIZE, FIG_WIDTH, FIG_HEIGHT,
    LEGEND_PARAM, ALPHA, GRID_ALPHA, MARKERS, YLABEL,
)
from utils.helper import get_run_info

ANNEAL_PERCENTS = [5, 10, 20, 50, 100]
SIZE = 60
TOKEN = TOKEN_LIST[SIZE][-1]  # 192

PERCENT_COLORS = {
    5: "grey",
    10: "forestgreen",
    20: "violet",
    50: "royalblue",
    100: "darkorange",
}
PERCENT_LABELS = {p: f"{p}%" for p in ANNEAL_PERCENTS}


# ── 2.1  Quantisation bar plot ──────────────────────────────────────────────

def plot_quant_bar(results):
    """Bar plot: 5 bars for anneal percentages, y = 4-bit quantised dclm_val."""

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.7, FIG_HEIGHT * 1.2))

    x = np.arange(len(ANNEAL_PERCENTS))
    bar_width = 0.55
    vals = []
    colors = []

    for ap in ANNEAL_PERCENTS:
        run_info = get_run_info(
            results, SIZE, "adamw",
            quantized=True, model_type="hf",
            anneal=True, anneal_percent=ap, anneal_optim="adamw",
            pretrain_lrs="wsd2",
        )
        if run_info and run_info["quantized"].get(4, {}).get(TOKEN):
            vals.append(run_info["quantized"][4][TOKEN]["dclm_quant"])
        else:
            vals.append(None)
        colors.append(PERCENT_COLORS[ap])

    valid_vals = [v for v in vals if v is not None]

    ax.bar(
        x, vals, bar_width,
        color=colors,
        alpha=ALPHA,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([PERCENT_LABELS[p] for p in ANNEAL_PERCENTS], fontsize=FONTSIZE["TICKS"])
    ax.set_xlabel("Anneal Percentage", fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(YLABEL["PT_LOSS_QUANTIZED"], fontsize=FONTSIZE["AXIS"])
    ax.tick_params(axis="y", labelsize=FONTSIZE["TICKS"])
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_title("4-bit Quantisation", fontsize=FONTSIZE["TITLE"])

    # ylim ±5% of min/max
    if valid_vals:
        y_min, y_max = min(valid_vals), max(valid_vals)
        ax.set_ylim(y_min * 0.95, y_max * 1.05)

    plt.tight_layout()
    out_dir = os.path.join(RESULTS_DIR, "plots/anneal_percent")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "quant_bar.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved quant_bar.pdf")


# ── 2.2  Perturbation line plot ─────────────────────────────────────────────

def plot_perturbation(results):
    """Line plot: dclm_val (y) vs increasing perturbation (x), 5 lines for anneal %."""

    perturbations = sorted(PERTURBATIONS)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.7, FIG_HEIGHT * 1.2))

    for i, ap in enumerate(ANNEAL_PERCENTS):
        run_info = get_run_info(
            results, SIZE, "adamw",
            perturb=True,
            anneal=True, anneal_percent=ap, anneal_optim="adamw",
            pretrain_lrs="wsd2",
        )
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
                label=PERCENT_LABELS[ap],
                marker=MARKERS[i % len(MARKERS)],
                color=PERCENT_COLORS[ap],
                alpha=ALPHA,
            )

    ax.set_xlabel("Perturbation (γ)", fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(YLABEL["PT_LOSS_PERTURBED"], fontsize=FONTSIZE["AXIS"])
    ax.set_title("Perturbation", fontsize=FONTSIZE["TITLE"])
    ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"])
    ax.grid(True, alpha=GRID_ALPHA)

    # Horizontal legend at bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Anneal %",
        fontsize=FONTSIZE["LEGEND"],
        title_fontsize=FONTSIZE["LEGEND"],
        loc=LEGEND_PARAM["LOC"],
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(handles),
    )

    plt.tight_layout()
    out_dir = os.path.join(RESULTS_DIR, "plots/anneal_percent")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "perturbation.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved perturbation.pdf")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as f:
        results = json.load(f)

    out_dir = os.path.join(RESULTS_DIR, "plots/anneal_percent")
    os.makedirs(out_dir, exist_ok=True)

    plot_quant_bar(results)
    plot_perturbation(results)


if __name__ == "__main__":
    main()
