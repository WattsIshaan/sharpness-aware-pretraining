"""
Quantization plots: OLMES eval score (from ModelEvaluationDownstreamOLMo) vs rho.
Plot 1–2: Pretrain 4T, Midtrain 5B — SAM, 4-SAM, SAM-Anneal (lines) + AdamW baseline + skyline (max without quant).
Plot 3–4: Pretrain 4T, Midtrain 50B — Unquantized baselines (hlines) + 4-bit bars; degradation arrows; SAM vs AdamW callout.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from utils.config_globals import RESULTS_DIR
from utils.plotting_globals import FONTSIZE, ALPHA, GRID_ALPHA, COLOR_MAP
from preprocess_midtrain_olmo import (
    parse_results,
    get_olmo_run_info,
    OLMES_EVAL_TASKS,
    OLMES_EVAL_GROUPS,
)

# Rho values to scan
RHOS = [2e-2, 5e-2, 1e-1, 1.25e-1, 1.5e-1, 2e-1, 3e-1, 4e-1, 5e-1, 1e0]
ANNEAL_RHOS = [5e-2, 1e-1, 1.5e-1, 2e-1]
MSAM_RHOS = [5e-2, 1e-1, 1.5e-1, 2e-1]


def _rho_label(rho):
    coeff, exp = f"{rho:.2e}".replace("+", "").split("e")
    coeff = coeff.rstrip("0").rstrip(".")
    return f"{coeff}e{exp}"


def _avg_score(results, eval_type, pretrain_token, midtrain_tokens, **kwargs):
    """Average across all OLMES eval tasks for a config (uses olmes_eval, not task_metrics)."""
    matched = get_olmo_run_info(
        results,
        eval_type=eval_type,
        pretrain_token=pretrain_token,
        midtrain_tokens=midtrain_tokens,
        **kwargs,
    )
    if not matched:
        return None
    # Use first run that has olmes_eval (not all runs have it if OLMES file missing)
    for run in matched:
        oe = run.get("olmes_eval") or {}
        vals = [oe[t] for t in OLMES_EVAL_TASKS if t in oe and oe[t] is not None]
        if vals:
            return float(np.mean(vals))
    return None


def _avg_score_group(results, eval_type, pretrain_token, midtrain_tokens, task_list, **kwargs):
    """Average over task_list (OLMES task names) for a config; uses olmes_eval."""
    matched = get_olmo_run_info(
        results,
        eval_type=eval_type,
        pretrain_token=pretrain_token,
        midtrain_tokens=midtrain_tokens,
        **kwargs,
    )
    if not matched:
        return None
    for run in matched:
        oe = run.get("olmes_eval") or {}
        vals = [oe[t] for t in task_list if t in oe and oe[t] is not None]
        if vals:
            return float(np.mean(vals))
    return None


def _plot_quant_5b_avg(results, out_path):
    """Plot 1: 4T, 5B. Avg eval after quant vs rho; SAM, 4-SAM, SAM-Anneal; AdamW baseline; skyline (max without quant)."""
    base_kw = dict(pretrain_token=4, midtrain_tokens=5, batch_size=1024)

    # After quantization: use 4-bit downstream evals
    adamw_quant = _avg_score(results, eval_type="hf-4bit", optim="adamw", **base_kw)
    sam_quant = [_avg_score(results, eval_type="hf-4bit", optim="sam", rho=r, **base_kw) for r in RHOS]
    anneal_quant = [_avg_score(results, eval_type="hf-4bit", optim="sam", rho=r, anneal_sam=True, **base_kw) for r in ANNEAL_RHOS]
    msam_quant = [_avg_score(results, eval_type="hf-4bit", optim="sam", rho=r, per_microbatch=True, **base_kw) for r in MSAM_RHOS]

    # Without quantization: max over olmo and hf (and all techniques) for skyline
    candidates = []
    for eval_type in ("olmo", "hf"):
        v = _avg_score(results, eval_type=eval_type, optim="adamw", **base_kw)
        if v is not None:
            candidates.append(v)
        for r in RHOS:
            v = _avg_score(results, eval_type=eval_type, optim="sam", rho=r, **base_kw)
            if v is not None:
                candidates.append(v)
        for r in ANNEAL_RHOS:
            v = _avg_score(results, eval_type=eval_type, optim="sam", rho=r, anneal_sam=True, **base_kw)
            if v is not None:
                candidates.append(v)
        for r in MSAM_RHOS:
            v = _avg_score(results, eval_type=eval_type, optim="sam", rho=r, per_microbatch=True, **base_kw)
            if v is not None:
                candidates.append(v)
    skyline = max(candidates) if candidates else None

    has_any = (
        adamw_quant is not None
        or any(sam_quant)
        or any(anneal_quant)
        or any(msam_quant)
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if not has_any:
        ax.text(0.5, 0.5, "No data for this configuration\n(4T pretrain, 5B midtrain)", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        rho_labels = [_rho_label(r) for r in RHOS]
        anneal_labels = [_rho_label(r) for r in ANNEAL_RHOS]
        msam_labels = [_rho_label(r) for r in MSAM_RHOS]

        valid = [(l, v) for l, v in zip(rho_labels, sam_quant) if v is not None]
        if valid:
            xl, yl = zip(*valid)
            ax.plot(xl, yl, marker="o", color=COLOR_MAP["sam"], label="SAM", alpha=ALPHA, linewidth=2)
        valid_a = [(l, v) for l, v in zip(anneal_labels, anneal_quant) if v is not None]
        if valid_a:
            xl, yl = zip(*valid_a)
            ax.plot(xl, yl, marker="s", color="forestgreen", label="SAM-Anneal", alpha=ALPHA, linewidth=2, linestyle="--")
        valid_m = [(l, v) for l, v in zip(msam_labels, msam_quant) if v is not None]
        if valid_m:
            xl, yl = zip(*valid_m)
            ax.plot(xl, yl, marker="^", color="darkorchid", label="4-SAM", alpha=ALPHA, linewidth=2)
        if adamw_quant is not None:
            ax.axhline(adamw_quant, color=COLOR_MAP["adamw"], linestyle=":", linewidth=2, label="AdamW")
        if skyline is not None:
            ax.axhline(skyline, color="gray", linestyle="-", linewidth=1.5, label="Skyline (no quant)")

        ax.set_xlabel("SAM rho", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Avg eval score (after 4bit quant)", fontsize=FONTSIZE["AXIS"])
        ax.set_title("4T pretrain, 5B midtrain — after 4bit quantization", fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(fontsize=FONTSIZE["LEGEND"] - 1, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_quant_5b_group(results, out_path):
    """Plot 2: Same as Plot 1 but 1x3 subplots for the 3 OLMES groups (mcq, generative, heldout)."""
    extra_kw = dict(batch_size=1024)
    group_names = list(OLMES_EVAL_GROUPS.keys())
    task_groups = [OLMES_EVAL_GROUPS[g] for g in group_names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for col, (task_list, gname) in enumerate(zip(task_groups, group_names)):
        ax = axes[col]
        adamw_quant = _avg_score_group(results, "hf-4bit", 4, 5, task_list, optim="adamw", **extra_kw)
        sam_quant = [_avg_score_group(results, "hf-4bit", 4, 5, task_list, optim="sam", rho=r, **extra_kw) for r in RHOS]
        anneal_quant = [_avg_score_group(results, "hf-4bit", 4, 5, task_list, optim="sam", rho=r, anneal_sam=True, **extra_kw) for r in ANNEAL_RHOS]
        msam_quant = [_avg_score_group(results, "hf-4bit", 4, 5, task_list, optim="sam", rho=r, per_microbatch=True, **extra_kw) for r in MSAM_RHOS]
        candidates = []
        for et in ("olmo", "hf"):
            v = _avg_score_group(results, et, 4, 5, task_list, optim="adamw", **extra_kw)
            if v is not None:
                candidates.append(v)
            for r in RHOS:
                v = _avg_score_group(results, et, 4, 5, task_list, optim="sam", rho=r, **extra_kw)
                if v is not None:
                    candidates.append(v)
        skyline = max(candidates) if candidates else None

        rho_labels = [_rho_label(r) for r in RHOS]
        anneal_labels = [_rho_label(r) for r in ANNEAL_RHOS]
        msam_labels = [_rho_label(r) for r in MSAM_RHOS]
        valid = [(l, v) for l, v in zip(rho_labels, sam_quant) if v is not None]
        if valid:
            xl, yl = zip(*valid)
            ax.plot(xl, yl, marker="o", color=COLOR_MAP["sam"], label="SAM", alpha=ALPHA, linewidth=2)
        valid_a = [(l, v) for l, v in zip(anneal_labels, anneal_quant) if v is not None]
        if valid_a:
            xl, yl = zip(*valid_a)
            ax.plot(xl, yl, marker="s", color="forestgreen", label="SAM-Anneal", alpha=ALPHA, linewidth=2, linestyle="--")
        valid_m = [(l, v) for l, v in zip(msam_labels, msam_quant) if v is not None]
        if valid_m:
            xl, yl = zip(*valid_m)
            ax.plot(xl, yl, marker="^", color="darkorchid", label="4-SAM", alpha=ALPHA, linewidth=2)
        if adamw_quant is not None:
            ax.axhline(adamw_quant, color=COLOR_MAP["adamw"], linestyle=":", linewidth=2, label="AdamW")
        if skyline is not None:
            ax.axhline(skyline, color="gray", linestyle="-", linewidth=1.5, label="Skyline (no quant)")
        ax.set_xlabel("SAM rho", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Avg score", fontsize=FONTSIZE["AXIS"])
        ax.set_title(gname, fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=GRID_ALPHA)
        if col == 0:
            ax.legend(fontsize=FONTSIZE["LEGEND"] - 2, loc="best")
    fig.suptitle("4T pretrain, 5B midtrain — group-wise (after quantization)", fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _pct_degradation(baseline, quantized):
    """Percent drop from unquantized to 4-bit: 100 * (baseline - q) / baseline."""
    if baseline is None or quantized is None or baseline == 0:
        return None
    return 100.0 * (baseline - quantized) / baseline


def _plot_quant_50b_bar(results, out_path):
    """4T, 50B, AdamW vs SAM (ρ=5e-2, ρ=1e-1). H-lines = unquantized OLMES avg; bars = 4-bit avg; arrows = % degradation."""
    base_kw = dict(pretrain_token=4, midtrain_tokens=50, batch_size=1024)
    sam_rhos = (5e-2,)

    adamw_quant = _avg_score(results, eval_type="hf-4bit", optim="adamw", **base_kw)
    sam_quant_by_rho = {
        rho: _avg_score(results, eval_type="hf-4bit", optim="sam", rho=rho, **base_kw) for rho in sam_rhos
    }

    adamw_noq = _avg_score(results, eval_type="olmo", optim="adamw", **base_kw)
    sam_noq_by_rho = {
        rho: _avg_score(results, eval_type="olmo", optim="sam", rho=rho, **base_kw) for rho in sam_rhos
    }
    for et in ("hf",):
        v = _avg_score(results, eval_type=et, optim="adamw", **base_kw)
        if v is not None:
            adamw_noq = v if adamw_noq is None else max(adamw_noq, v)
        for rho in sam_rhos:
            v = _avg_score(results, eval_type=et, optim="sam", rho=rho, **base_kw)
            if v is not None:
                prev = sam_noq_by_rho.get(rho)
                sam_noq_by_rho[rho] = v if prev is None else max(prev, v)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    has_any = (
        adamw_quant is not None
        or any(v is not None for v in sam_quant_by_rho.values())
        or adamw_noq is not None
        or any(v is not None for v in sam_noq_by_rho.values())
    )
    if not has_any:
        ax.text(
            0.5,
            0.5,
            "No data for this configuration\n(4T pretrain, 50B midtrain)",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        x = np.array([0.0, 1.0])
        bar_w = 0.52
        h_adam = adamw_quant if adamw_quant is not None else 0.0
        h_sam_5e2 = sam_quant_by_rho.get(5e-2) if sam_quant_by_rho.get(5e-2) is not None else 0.0
        # h_sam_1e1 = sam_quant_by_rho.get(1e-1) if sam_quant_by_rho.get(1e-1) is not None else 0.0
        ax.bar(
            x[0],
            h_adam,
            width=bar_w,
            color=COLOR_MAP["adamw"],
            alpha=ALPHA,
            edgecolor="black",
            linewidth=0.6,
            zorder=2,
        )
        ax.bar(
            x[1],
            h_sam_5e2,
            width=bar_w,
            color=COLOR_MAP["sam"],
            alpha=ALPHA,
            edgecolor="black",
            linewidth=0.6,  
            zorder=2,
        )
        # ax.bar(
        #     x[2],
        #     h_sam_1e1,
        #     width=bar_w,
        #     color=COLOR_MAP["sam_rho1e-1"],
        #     alpha=ALPHA,
        #     edgecolor="black",
        #     linewidth=0.6,
        #     zorder=2,
        # )

        if adamw_noq is not None:
            ax.axhline(
                adamw_noq,
                color=COLOR_MAP["adamw"],
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                zorder=1,
            )
        sam_noq_5e2 = sam_noq_by_rho.get(5e-2)
        if sam_noq_5e2 is not None:
            ax.axhline(
                sam_noq_5e2,
                color=COLOR_MAP["sam"],
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                zorder=1,
            )
        # sam_noq_1e1 = sam_noq_by_rho.get(1e-1)
        # if sam_noq_1e1 is not None:
        #     ax.axhline(
        #         sam_noq_1e1,
        #         color=COLOR_MAP["sam_rho1e-1"],
        #         linestyle="--",
        #         linewidth=2,
        #         alpha=0.9,
        #         zorder=1,
        #     )

        for xi, nq, q, c in zip(
            x,
            [adamw_noq, sam_noq_5e2],
            [adamw_quant, sam_quant_by_rho.get(5e-2)],
            [COLOR_MAP["adamw"], COLOR_MAP["sam"]],
        ):
            if nq is None or q is None:
                continue
            ax.annotate(
                "",
                xy=(xi, q),
                xytext=(xi, nq),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    lw=2.2,
                    shrinkA=4,
                    shrinkB=4,
                ),
                zorder=4,
            )
            pd = _pct_degradation(nq, q)
            if pd is not None:
                ym = 0.5 * (nq + q)
                ax.text(
                    xi + 0.10,
                    ym,
                    f"{pd:.1f}%",
                    fontsize=FONTSIZE["TICKS"] - 1,
                    va="center",
                    ha="left",
                    color="black",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(["AdamW", "SAM (ρ=5e-2)"], fontsize=FONTSIZE["TICKS"] - 1)
        ax.set_ylabel("Average PT Eval Score", fontsize=FONTSIZE["AXIS"])
        ax.set_title(
            "Degradation after 4-bit Quantization",
            fontsize=FONTSIZE["TITLE"],
        )
        ax.tick_params(axis="y", labelsize=FONTSIZE["TICKS"] - 1)
        ax.grid(True, alpha=GRID_ALPHA, axis="y", zorder=0)

        ys = [v for v in (adamw_quant, adamw_noq, sam_quant_by_rho.get(5e-2), sam_noq_5e2) if v is not None]
        if ys:
            lo, hi = min(ys), max(ys)
            pad = max((hi - lo) * 0.15, 0.5)
            ax.set_ylim(lo - pad, hi + pad)

        # ax.legend(fontsize=FONTSIZE["LEGEND"] - 2, loc="upper left")

        sam_quant_5e2 = sam_quant_by_rho.get(5e-2)
        if (adamw_quant is not None and sam_quant_5e2 is not None and
            adamw_noq is not None and sam_noq_5e2 is not None):

            degradation_adamw = 100.0 * (adamw_noq - adamw_quant) / adamw_noq if adamw_noq else None
            degradation_sam = 100.0 * (sam_noq_5e2 - sam_quant_5e2) / sam_noq_5e2 if sam_noq_5e2 else None

            if degradation_adamw is not None and degradation_sam is not None and degradation_adamw != 0:
                rel_improvement = (degradation_adamw - degradation_sam) / degradation_adamw * 100
                box_txt = f"SAM (ρ=5e-2) reduces PTQ degradation by {rel_improvement:.1f}%"
                ax.text(
                    0.02,
                    0.96,
                    box_txt,
                    transform=ax.transAxes,
                    fontsize=FONTSIZE["TICKS"] - 1,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.4", alpha=0.92),
                    zorder=6,
                )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_quant_50b_group_bar(results, out_path):
    """Plot 4 (group): Same as bar but 1x3 for the 3 OLMES groups (mcq, generative, heldout)."""
    extra_kw = dict(batch_size=1024)
    group_names = list(OLMES_EVAL_GROUPS.keys())
    task_groups = [OLMES_EVAL_GROUPS[g] for g in group_names]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for col, (task_list, gname) in enumerate(zip(task_groups, group_names)):
        ax = axes[col]
        adamw_quant = _avg_score_group(results, "hf-4bit", 4, 50, task_list, optim="adamw", **extra_kw)
        sam_quant_5e2 = _avg_score_group(results, "hf-4bit", 4, 50, task_list, optim="sam", rho=5e-2, **extra_kw)
        sam_quant_1e1 = _avg_score_group(results, "hf-4bit", 4, 50, task_list, optim="sam", rho=1e-1, **extra_kw)

        # Non-quantized skyline: max across non-quant eval types (olmo and hf)
        adamw_noq = _avg_score_group(results, "olmo", 4, 50, task_list, optim="adamw", **extra_kw)
        sam_noq_5e2 = _avg_score_group(results, "olmo", 4, 50, task_list, optim="sam", rho=5e-2, **extra_kw)
        sam_noq_1e1 = _avg_score_group(results, "olmo", 4, 50, task_list, optim="sam", rho=1e-1, **extra_kw)
        for et in ("hf",):
            v = _avg_score_group(results, et, 4, 50, task_list, optim="adamw", **extra_kw)
            if v is not None:
                adamw_noq = v if adamw_noq is None else max(adamw_noq, v)
            v = _avg_score_group(results, et, 4, 50, task_list, optim="sam", rho=5e-2, **extra_kw)
            if v is not None:
                sam_noq_5e2 = v if sam_noq_5e2 is None else max(sam_noq_5e2, v)
            v = _avg_score_group(results, et, 4, 50, task_list, optim="sam", rho=1e-1, **extra_kw)
            if v is not None:
                sam_noq_1e1 = v if sam_noq_1e1 is None else max(sam_noq_1e1, v)

        skyline = max((x for x in (adamw_noq, sam_noq_5e2, sam_noq_1e1) if x is not None), default=None)
        ax.bar(
            ["AdamW", "SAM (ρ=5e-2)", "SAM (ρ=1e-1)"],
            [adamw_quant or 0, sam_quant_5e2 or 0, sam_quant_1e1 or 0],
            color=[COLOR_MAP["adamw"], COLOR_MAP["sam"], COLOR_MAP["sam_rho1e-1"]],
            alpha=ALPHA,
        )
        if skyline is not None:
            ax.axhline(skyline, color="gray", linestyle="-", linewidth=1.5, label="Skyline (no quant)")
        ax.set_ylabel("Avg score", fontsize=FONTSIZE["AXIS"])
        ax.set_title(gname, fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.grid(True, alpha=GRID_ALPHA, axis="y")
    fig.suptitle("4T pretrain, 50B midtrain — group-wise (after quantization)", fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    results = parse_results()
    out_dir = os.path.join(RESULTS_DIR, "plots", "quantization")
    os.makedirs(out_dir, exist_ok=True)

    # _plot_quant_5b_avg(results, os.path.join(out_dir, "quant_olmes_avg_4T_5B.png"))
    # _plot_quant_5b_group(results, os.path.join(out_dir, "quant_olmes_group_4T_5B.png"))
    _plot_quant_50b_bar(results, os.path.join(out_dir, "quantization_1b.png"))
    # _plot_quant_50b_group_bar(results, os.path.join(out_dir, "quant_olmes_group_4T_50B.png"))


if __name__ == "__main__":
    main()
