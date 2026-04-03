"""AdamW Pareto on StarCoder vs DCLM: cosine PT LR 6e-4 vs WSD PT LR 3e-4 (60M, 192B tokens)."""

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
    XLABEL,
    YLABEL,
)

SIZE = 60
PRETRAIN_TOKEN = 192
CPT_DATASET = "starcoder"
CPT_TOKENS = 10
OPTIM = "adamw"

PT_LR_COSINE = 3e-4
PT_LR_WSD = 3e-4
SCHED_COSINE = "cosine"
SCHED_WSD = "wsd2"

WSD_ANNEAL_PERCENT = 100
WSD_ANNEAL_OPTIM = "adamw"
WSD_ANNEAL_MATCH = "token"

# pareto.py limits: 60M + starcoder
XLIM = (0, 4.5)
YLIM = (0, 2.8)


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


def plot_series(
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path. Default: results/plots/lr_matched/adamw_60m_starcoder_192b_cos6e4_wsd3e4.png",
    )
    args = parser.parse_args()
    out = args.out or os.path.join(
        RESULTS_DIR,
        "plots/lr_matched/adamw_60m_starcoder_192b_cos6e4_linear3e4.png",
    )

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    cosine_info = get_run_info(
        results,
        SIZE,
        OPTIM,
        CPT_DATASET,
        cpt_tokens=CPT_TOKENS,
        use_ewc=False,
        pt_lr=PT_LR_COSINE,
        pretrain_lrs=SCHED_COSINE,
    )
    x_c, y_c = collect_pareto_points(cosine_info, PRETRAIN_TOKEN, CPT_DATASET)
    plot_series(
        ax,
        x_c,
        y_c,
        COLOR_MAP["cosine_adamw"],
        MARKERS[0],
        "-",
        f"Cosine, PT LR={PT_LR_COSINE:g}",
    )

    wsd_info = get_run_info(
        results,
        SIZE,
        OPTIM,
        CPT_DATASET,
        cpt_tokens=CPT_TOKENS,
        use_ewc=False,
        pt_lr=PT_LR_WSD,
        anneal=True,
        anneal_percent=WSD_ANNEAL_PERCENT,
        anneal_optim=WSD_ANNEAL_OPTIM,
        anneal_match=WSD_ANNEAL_MATCH,
        pretrain_lrs=SCHED_WSD,
    )
    x_w, y_w = collect_pareto_points(wsd_info, PRETRAIN_TOKEN, CPT_DATASET)
    plot_series(
        ax,
        x_w,
        y_w,
        COLOR_MAP["wsd_adamw"],
        MARKERS[1],
        "-",
        f"Linear, PT LR={PT_LR_WSD:g}",
    )

    ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
    ax.set_title(
        f"{SIZE}M · {CPT_DATASET_MAP[CPT_DATASET]} · {PRETRAIN_TOKEN}B",
        fontsize=FONTSIZE["TITLE"],
    )
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE["TICKS"])
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc=LEGEND_PARAM["LOC"],
            bbox_to_anchor=(0.5, -0.30),
            ncol=1,
            fontsize=FONTSIZE["LEGEND"],
        )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
