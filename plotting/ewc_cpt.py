"""Pareto front: CPT on StarCoder vs pretrain loss (DCLM), 60M @ 192B tokens.

Baseline AdamW/SAM (solid hull) and EWC (dashed hull, same colors/markers as pareto.py).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from utils.config_globals import RESULTS_DIR
from utils.helper import get_run_info
from utils.plotting_globals import (
    ALPHA,
    COLOR_MAP,
    CPT_DATASET_MAP,
    FIG_HEIGHT,
    FIG_WIDTH,
    FONTSIZE,
    LEGEND_PARAM,
    MARKERS,
    OPTIM_MAP,
    XLABEL,
    YLABEL,
)

SIZE = 60
PRETRAIN_TOKEN = 192  # 192B tokens (60M model)
CPT_DATASET = "starcoder"
# CPT_DATASET = "musicpile"
CPT_TOKENS = 10
# Stored in final_results.json for EWC runs (run_type "ewc")
EWC_LAMBDA = 4000.0
EWC_BS = 100

# Match pareto.py limits for 60M + starcoder
XLIM = (0, 4.5)
YLIM = (0, 2.8)


def _format_cpt_lr(lr: float) -> str:
    return f"{lr:g}"


def collect_pareto_points(
    run_info, pt_token: int
) -> tuple[list[float], list[float], list[float]]:
    if run_info is None or not run_info.get("cpt"):
        return [], [], []
    dclm_val: list[float] = []
    cpt_val: list[float] = []
    cpt_lrs: list[float] = []
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
                    y_val = cell[pt_token][CPT_DATASET]
                except (KeyError, TypeError):
                    continue
                if x0 <= x_val <= x1 and y0 <= y_val <= y1:
                    dclm_val.append(x_val)
                    cpt_val.append(y_val)
                    cpt_lrs.append(float(cell[pt_token]["cpt_lr"]))
    return dclm_val, cpt_val, cpt_lrs


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


LR_ANNOT_FONTSIZE = max(6, FONTSIZE["TICKS"] - 6)


def plot_series(
    ax,
    dclm_val: list[float],
    cpt_val: list[float],
    cpt_lrs: list[float],
    color: str,
    marker: str,
    linestyle: str,
    label: str | None,
) -> None:
    if not dclm_val:
        return
    ax.scatter(dclm_val, cpt_val, color=color, alpha=ALPHA, marker=marker)
    # for x, y, lr in zip(dclm_val, cpt_val, cpt_lrs):
    #     ax.annotate(
    #         _format_cpt_lr(lr),
    #         (x, y),
    #         xytext=(4, 4),
    #         textcoords="offset points",
    #         fontsize=LR_ANNOT_FONTSIZE,
    #         color=color,
    #         alpha=min(1.0, ALPHA + 0.05),
    #     )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=f"Output path (.pdf). Default: results/plots/ewc_cpt/pareto_60m_{CPT_DATASET}_192b.png",
    )
    args = parser.parse_args()
    out = args.out or os.path.join(
        RESULTS_DIR, f"plots/ewc_cpt/pareto_60m_{CPT_DATASET}_192b.png"
    )

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    optim_list = ["adamw", "sam"]
    for idx, optim in enumerate(optim_list):
        marker = MARKERS[idx]
        color = COLOR_MAP[optim]

        base = get_run_info(
            results, SIZE, optim, CPT_DATASET, cpt_tokens=CPT_TOKENS, use_ewc=False
        )
        xd, yd, ld = collect_pareto_points(base, PRETRAIN_TOKEN)
        plot_series(
            ax,
            xd,
            yd,
            ld,
            color,
            marker,
            "-",
            OPTIM_MAP[optim],
        )

        ewc = get_run_info(
            results,
            SIZE,
            optim,
            CPT_DATASET,
            cpt_tokens=CPT_TOKENS,
            use_ewc=True,
            ewc_lambda=EWC_LAMBDA,
            ewc_bs=EWC_BS,
        )
        xe, ye, le = collect_pareto_points(ewc, PRETRAIN_TOKEN)
        plot_series(
            ax,
            xe,
            ye,
            le,
            color,
            marker,
            "--",
            f"{OPTIM_MAP[optim]} + EWC",
        )

    ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
    ax.set_title(
        f"{SIZE}M · {CPT_DATASET_MAP[CPT_DATASET]} · {PRETRAIN_TOKEN}B PT Tokens",
        fontsize=FONTSIZE["TITLE"],
    )
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE["TICKS"])
    ax.grid(True)
    # ax.set_xlim(XLIM[0], XLIM[1])
    # ax.set_ylim(YLIM[0], YLIM[1])

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc=LEGEND_PARAM["LOC"],
            bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"],
            ncol=2,
            fontsize=FONTSIZE["LEGEND"],
        )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
