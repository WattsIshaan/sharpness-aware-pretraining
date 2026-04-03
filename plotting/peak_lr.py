import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from utils.config_globals import RESULTS_DIR, PERTURBATIONS, TOKEN_LIST
from utils.plotting_globals import (
    ALPHA,
    COLOR_MAP,
    CPT_DATASET_MAP,
    FONTSIZE,
    FIG_HEIGHT,
    FIG_WIDTH,
    GRID_ALPHA,
    LEGEND_PARAM,
    MARKERS,
    XLABEL,
    YLABEL,
)
from utils.helper import get_run_info

PEAK_LRS = [1e-4, 3e-4, 6e-4]
LR_LABELS = {1e-4: "1e-4", 3e-4: "3e-4", 6e-4: "6e-4"}
LR_COLORS = {1e-4: COLOR_MAP[1e-4], 3e-4: COLOR_MAP[3e-4], 6e-4: COLOR_MAP[6e-4]}
SIZE = 60
PRETRAIN_TOKEN = 192  # 60M checkpoint list ends at 192B
TOKEN = TOKEN_LIST[SIZE][-1]  # 192 — same as PRETRAIN_TOKEN

CPT_DATASET = "starcoder"
CPT_TOKENS = 10
# pareto.py: 60M + starcoder
XLIM = (0, 5)
YLIM = (0, 4)

WSD_ANNEAL_PERCENT = 10
WSD_ANNEAL_OPTIM = "adamw"
WSD_ANNEAL_MATCH = "token"
SCHED_COSINE = "cosine"
SCHED_WSD = "wsd"

# Perturbation plot: cap y-axis; artists clip at the axes patch (default clip_on).
PERTURB_PLOT_Y_MAX = 5.0


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
    plt.savefig(os.path.join(out_dir, "quant_bar.png"), bbox_inches="tight")
    plt.close()
    print("Saved quant_bar.png")


# ── 1.2  Pareto (1×2: Cosine | WSD), StarCoder only, 60M 192B — same logic as ewc_cpt.py ─────


def collect_pareto_points(
    run_info, pt_token: int, cpt_dataset: str
) -> tuple[list[float], list[float]]:
    if run_info is None or not run_info.get("cpt"):
        return [], []
    dclm_val: list[float] = []
    cpt_val: list[float] = []
    x0, x1 = XLIM
    y0, y1 = YLIM
    for cpt_lr in sorted(run_info["cpt"].keys()):
        for cpt_wd in sorted(run_info["cpt"][cpt_lr].keys()):
            for cpt_bs in sorted(run_info["cpt"][cpt_lr][cpt_wd].keys()):
                try:
                    cell = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs]
                    if pt_token not in cell:
                        continue
                    x_val = cell[pt_token]["dclm_val"]
                    y_val = cell[pt_token][cpt_dataset]
                except (KeyError, TypeError):
                    continue
                if x0 <= x_val <= x1 and y0 <= y_val <= y1:
                    dclm_val.append(x_val)
                    cpt_val.append(y_val)
    return dclm_val, cpt_val


def hull_polyline(dclm_val: list[float], cpt_val: list[float]) -> np.ndarray | None:
    if len(dclm_val) <= 2:
        return None
    points = np.empty((len(dclm_val), 2))
    points[:, 0] = dclm_val
    points[:, 1] = cpt_val
    hull = ConvexHull(points)
    h = np.append(hull.vertices, hull.vertices[0])
    new_points = np.empty((len(h), 2))
    new_points[:, 0] = points[h, 0]
    new_points[:, 1] = points[h, 1]
    min_idx = np.argmin(new_points[:, 0])
    new_points = np.roll(new_points, -min_idx, axis=0)
    for i in range(len(new_points) - 1):
        if new_points[i][0] > new_points[i + 1][0]:
            new_points = new_points[: i + 1, :]
            break
    return new_points


def _plot_pareto_series(
    ax,
    dclm_val: list[float],
    cpt_val: list[float],
    color: str,
    marker: str,
    linestyle: str,
    label: str | None,
) -> None:
    if not dclm_val:
        return
    ax.scatter(dclm_val, cpt_val, color=color, alpha=ALPHA, marker=marker)
    line = hull_polyline(dclm_val, cpt_val)
    if line is not None:
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=color,
            linestyle=linestyle,
            label=label,
            marker=marker,
            alpha=ALPHA,
        )
    elif label:
        ax.plot([], [], color=color, linestyle=linestyle, label=label, marker=marker)


def plot_pareto(results):
    """1×2 Pareto: DCLM vs StarCoder CPT loss at 60M / 192B; left Cosine, right WSD (peak LRs 1e-4, 3e-4, 6e-4)."""
    out_dir = os.path.join(RESULTS_DIR, "plots/peak_lr")
    os.makedirs(out_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.2), sharey=True, sharex=True)

    for i, lr in enumerate(PEAK_LRS):
        run_info = get_run_info(
            results,
            SIZE,
            "adamw",
            CPT_DATASET,
            cpt_tokens=CPT_TOKENS,
            use_ewc=False,
            pt_lr=lr,
            pretrain_lrs=SCHED_COSINE,
        )
        xd, yd = collect_pareto_points(run_info, PRETRAIN_TOKEN, CPT_DATASET)
        _plot_pareto_series(
            axs[0],
            xd,
            yd,
            LR_COLORS[lr],
            MARKERS[i % len(MARKERS)],
            "-",
            f"PT LR={LR_LABELS[lr]}",
        )

    for i, lr in enumerate(PEAK_LRS):
        run_info = get_run_info(
            results,
            SIZE,
            "adamw",
            CPT_DATASET,
            cpt_tokens=CPT_TOKENS,
            use_ewc=False,
            pt_lr=lr,
            anneal=True,
            anneal_percent=WSD_ANNEAL_PERCENT,
            anneal_optim=WSD_ANNEAL_OPTIM,
            anneal_match=WSD_ANNEAL_MATCH,
            pretrain_lrs=SCHED_WSD,
        )
        xw, yw = collect_pareto_points(run_info, PRETRAIN_TOKEN, CPT_DATASET)
        _plot_pareto_series(
            axs[1],
            xw,
            yw,
            LR_COLORS[lr],
            MARKERS[i % len(MARKERS)],
            "-",
            f"PT LR={LR_LABELS[lr]}",
        )

    for ax, title in zip(axs, ("Cosine", "WSD")):
        ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
        ax.set_title(title, fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"])
        ax.grid(True, alpha=GRID_ALPHA)

    axs[0].set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
    fig.suptitle(
        f"{SIZE}M · {PRETRAIN_TOKEN}B · {CPT_DATASET_MAP[CPT_DATASET]}",
        fontsize=FONTSIZE["TITLE"] + 1,
        y=1.02,
    )
    handles, labels_leg = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_leg,
            loc=LEGEND_PARAM["LOC"],
            bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"],
            ncol=len(handles),
            fontsize=FONTSIZE["LEGEND"],
        )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pareto_starcoder.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── 1.3  Perturbation line plot (1×2 subplots: Cosine | WSD) ────────────────

def plot_perturbation(results):
    """Line plot: dclm_val (y) vs gamma (x), 3 lines per LR, 1×2 subplot for Cosine / WSD."""

    perturbations = sorted(PERTURBATIONS)

    schedules = [
        ("Cosine", dict(anneal=False)),
        ("WSD",    dict(anneal=True, anneal_percent=10, anneal_optim="adamw")),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.3, FIG_HEIGHT * 1.2), sharey=True)
    y_all: list[float] = []

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
                y_all.extend(y_vals)
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
        ax.set_axisbelow(True)

    if y_all:
        y_lo = min(y_all)
        pad = max(0.02, (PERTURB_PLOT_Y_MAX - y_lo) * 0.04)
        axs[0].set_ylim(y_lo - pad, PERTURB_PLOT_Y_MAX)
    else:
        axs[0].set_ylim(0.0, PERTURB_PLOT_Y_MAX)

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
    plt.savefig(os.path.join(out_dir, "perturbation.png"), bbox_inches="tight")
    plt.close()
    print("Saved perturbation.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as f:
        results = json.load(f)

    out_dir = os.path.join(RESULTS_DIR, "plots/peak_lr")
    os.makedirs(out_dir, exist_ok=True)

    plot_quant_bar(results)
    plot_perturbation(results)
    plot_pareto(results)


if __name__ == "__main__":
    main()
