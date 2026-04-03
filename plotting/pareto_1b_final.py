"""
One row of two Pareto-style panels:

  * Left: CPT on **meta-math** — GSM8K (y) vs mean OLMES eval score (x).
  * Right: CPT on **magicoder** — HumanEval / ``codex_humaneval`` (y) vs same x.

**x-axis** matches ``pareto_1b_thresholded.plot_thresholded_pareto_olmes_raw``: mean over
``OLMES_EVAL_TASKS`` from ``preprocess_midtrain_olmo`` (same task keys as the thresholded OLMES
plots).

Filters: 4T pretrain, 50B midtrain, AdamW vs SAM (ρ=5e−2), batch 1024, olmes_eval.

Optional **CPT LR window** per dataset: ``LR_RANGE_BY_DATASET`` (or ``--lr-meta-math`` /
``--lr-magicoder`` on the CLI) keeps only runs whose ``cpt_lr`` lies in ``[min, max]`` inclusive;
``None`` for a dataset means no LR filter.

Convex hull: ``pareto_1b._convex_hull_boundary`` — ``ConvexHull``, then the **long** open path along
the hull between the vertices with **minimum** and **maximum** CPT LR (same idea as other 1B Pareto
plots; appropriate when both x and y are “higher is better” unlike ``pareto.py``’s loss-on-x slice).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.config_globals import RESULTS_DIR
from utils.plotting_globals import FONTSIZE, ALPHA, GRID_ALPHA, COLOR_MAP
from preprocess_midtrain_olmo import OLMES_EVAL_TASKS, parse_results, get_cpt_run_info
from pareto_1b import ADAMW_COLOR, _convex_hull_boundary, _lr_label, _mean_olmes_tasks_if_complete

CPT_TOKENS_BY_DATASET = {
    "meta-math": 80,
    "magicoder": 50,
}

PRETRAIN_T = 4
MIDTRAIN_B = 50
SAM_RHO = 5e-2
BATCH_SIZE = 1024

# Inclusive CPT LR bounds per dataset; None = include all LRs (no filter).
LR_RANGE_BY_DATASET = {
    "meta-math": (2e-5, 6e-5),
    "magicoder": (8e-5, 4e-4),
}


def _in_cpt_lr_range(lr, lr_range):
    """If lr_range is (lo, hi), require lo <= cpt_lr <= hi; if None, pass all finite LRs."""
    if lr_range is None:
        return lr is not None
    lo, hi = lr_range
    if lr is None:
        return False
    return float(lo) <= float(lr) <= float(hi)


def _rho_label(rho):
    if rho is None:
        return ""
    s = f"{rho:.2e}".replace("+", "")
    c, e = s.split("e")
    c = c.rstrip("0").rstrip(".")
    return f"{c}e{e}"


def _avg_olmes_eval_tasks(oe):
    """
    Mean OLMES score over ``OLMES_EVAL_TASKS``. Returns ``None`` if any task is missing or ``None``
    (same strict aggregation as ``pareto_1b._mean_olmes_tasks_if_complete`` / series helpers).
    """
    return _mean_olmes_tasks_if_complete(oe, OLMES_EVAL_TASKS)


def _runs_adamw_sam(results, cpt_dataset, cpt_tokens):
    runs = get_cpt_run_info(
        results,
        midtrain_tokens=MIDTRAIN_B,
        cpt_dataset=cpt_dataset,
        cpt_tokens=cpt_tokens,
        pretrain_token=PRETRAIN_T,
    )
    if not runs:
        return []
    out = []
    for r in runs:
        if r.get("batch_size") != BATCH_SIZE:
            continue
        opt = r.get("optimizer")
        rho = r.get("rho")
        if opt == "adamw" and rho is None:
            out.append(r)
        elif opt == "sam" and rho is not None and np.isclose(rho, SAM_RHO):
            out.append(r)
    return out


def _series_xy(runs, learning_key, lr_range=None):
    """x = mean OLMES over ``OLMES_EVAL_TASKS``; y = ``learning_key`` score from olmes_eval."""
    subset = sorted(runs, key=lambda r: float(r.get("cpt_lr") or 0))
    xs, ys, lrs = [], [], []
    for r in subset:
        if not _in_cpt_lr_range(r.get("cpt_lr"), lr_range):
            continue
        oe = r.get("olmes_eval") or {}
        if learning_key not in oe or oe[learning_key] is None:
            continue
        xf = _avg_olmes_eval_tasks(oe)
        if xf is None:
            continue
        xs.append(xf)
        ys.append(float(oe[learning_key]))
        lrs.append(r.get("cpt_lr"))
    return (xs, ys, lrs) if xs else (None, None, None)


def _plot_series(ax, runs, learning_key, lr_range=None):
    techs = [("adamw", None), ("sam", SAM_RHO)]
    markers = ["o", "s"]
    for i, (optim, rho) in enumerate(techs):
        subset = [r for r in runs if r.get("optimizer") == optim]
        if optim == "adamw":
            subset = [r for r in subset if r.get("rho") is None]
        else:
            subset = [r for r in subset if r.get("rho") is not None and np.isclose(r.get("rho"), rho)]
        x_vals, y_vals, lr_vals = _series_xy(subset, learning_key, lr_range=lr_range)
        if not x_vals:
            continue
        label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
        color = COLOR_MAP[optim]
        ax.scatter(
            x_vals, y_vals, marker=markers[i % 2], color=color, label=label, alpha=ALPHA, s=55, zorder=5
        )
        for x, y, lr in zip(x_vals, y_vals, lr_vals):
            lab = _lr_label(lr)
            if lab:
                ax.annotate(
                    lab,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=FONTSIZE["TICKS"] - 3,
                    color=color,
                    alpha=0.95,
                    zorder=6,
                )
        lr_ok = lr_vals if len(lr_vals) == len(x_vals) and all(lr is not None for lr in lr_vals) else None
        xh, yh = _convex_hull_boundary(x_vals, y_vals, lr_ok)
        if len(xh) >= 2:
            ax.plot(xh, yh, color=color, linewidth=2.4, zorder=3, solid_capstyle="round")


def _figure_1x2(results, out_path, lr_range_by_dataset=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    lr_by = {**LR_RANGE_BY_DATASET, **(lr_range_by_dataset or {})}

    r_mm = _runs_adamw_sam(results, "meta-math", CPT_TOKENS_BY_DATASET["meta-math"])
    r_mg = _runs_adamw_sam(results, "magicoder", CPT_TOKENS_BY_DATASET["magicoder"])

    for ax, runs, lk, title, ylab, ds_key in (
        (axes[0], r_mm, "gsm8k", "SFT on Meta-Math", "GSM8K", "meta-math"),
        (axes[1], r_mg, "codex_humaneval", "SFT on Magicoder", "HumanEval", "magicoder"),
    ):
        if not runs:
            ax.text(0.5, 0.5, "No CPT data", ha="center", va="center", transform=ax.transAxes)
        else:
            _plot_series(ax, runs, lk, lr_range=lr_by.get(ds_key))
        ax.set_xlabel("Average OLMES eval score", fontsize=FONTSIZE["AXIS"] - 1)
        ax.set_ylabel(ylab, fontsize=FONTSIZE["AXIS"])
        ax.set_title(title, fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.invert_xaxis()
        ax.invert_yaxis()

    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break
    if handles:
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=FONTSIZE["LEGEND"] - 1
        )

    fig.suptitle(
        "Learning–forgetting Pareto (OLMES_EVAL_TASKS avg. vs finetuning task)",
        fontsize=FONTSIZE["TITLE"] + 1,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pareto 1x2: OLMES avg vs GSM8K / HumanEval (final).")
    parser.add_argument(
        "--lr-meta-math",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Inclusive CPT LR range for the Meta-Math / GSM8K panel (overrides LR_RANGE_BY_DATASET).",
    )
    parser.add_argument(
        "--lr-magicoder",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Inclusive CPT LR range for the Magicoder / HumanEval panel (overrides LR_RANGE_BY_DATASET).",
    )
    args = parser.parse_args()

    lr_override = {}
    if args.lr_meta_math is not None:
        lr_override["meta-math"] = (args.lr_meta_math[0], args.lr_meta_math[1])
    if args.lr_magicoder is not None:
        lr_override["magicoder"] = (args.lr_magicoder[0], args.lr_magicoder[1])

    results = parse_results()
    out_dir = os.path.join(RESULTS_DIR, "plots", "pareto_1b_final")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "pareto_1b_final")
    _figure_1x2(results, f"{base}.pdf", lr_range_by_dataset=lr_override)


if __name__ == "__main__":
    main()
