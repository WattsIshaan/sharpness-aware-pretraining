"""
1B CPT "thresholded" Pareto: mean OLMES eval score (x) vs **y**, with vertical reference lines at the
pre-CPT downstream OLMES average for AdamW and SAM.

**Loss variant** (``plot_thresholded_pareto_olmes_raw``): **y** = finetuning val loss; horizontal line
at ``(1 + loss_margin)`` × minimum AdamW CPT loss. Margins per dataset: ``LOSS_MARGIN_4T_50B_BY_DATASET``
(Tulu ``0.005``). Black arrows from AdamW/SAM baselines to the hull at that loss; SAM vs AdamW
forgetting labels and summary box.

``main`` generates 4T / 50B midtrain figures for Tulu, StackMathQA, Musicpile (50M CPT), Meta-Math
(80M), and Magicoder (50M): (1) mean-OLMES thresholded Pareto, (2) a **per-task** grid
(``plot_thresholded_pareto_olmes_per_task_grid``) with the same loss threshold; AdamW + SAM ρ=5e-2.

Run from this directory (``plotting/``) with ``conda activate forgetting2`` then
``PYTHONPATH=. python pareto_1b_thresholded.py`` (SciPy required for the hull).

Unlike pareto_1b._plot_pareto_olmes_eval_avg, the x-axis is the raw average OLMES score
(not relative forgetting %). The x-axis is reversed so higher scores lie toward the left,
consistent with other CPT Pareto figures.

CPT runs are restricted to **final** eval only (``step=-1``: no ``step4000`` in the filename), so
each learning rate appears once. The hull line uses ``pareto_1b._convex_hull_boundary``: ``ConvexHull``,
then the **long** open path along the hull between min- and max-**CPT LR** vertices (appropriate for
OLMES vs loss; unlike ``pareto.py``’s min-``x`` monotone chain).
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy

from utils.config_globals import RESULTS_DIR
from utils.plotting_globals import ALPHA, CPT_DATASET_MAP, FONTSIZE, GRID_ALPHA, COLOR_MAP
from preprocess_midtrain_olmo import OLMES_EVAL_GROUPS, OLMES_EVAL_TASKS, get_cpt_run_info, parse_results
from pareto_1b import (
    _baseline_olmes_avg_score,
    _convex_hull_boundary,
    _cpt_lr_window,
    _lr_label,
    _mean_olmes_tasks_if_complete,
    _rho_label,
    _series_for_technique_olmes_avg,
    _series_for_technique_olmes_single_task,
)


def _hull_xy_with_lr(x_vals, y_vals, lr_vals):
    """``_convex_hull_boundary`` with CPT LR for the open hull path when ``lr_vals`` aligns with points."""
    lr_ok = (
        lr_vals
        if lr_vals is not None
        and len(lr_vals) == len(x_vals)
        and all(lr is not None for lr in lr_vals)
        else None
    )
    return _convex_hull_boundary(x_vals, y_vals, lr_ok)


def _loss_threshold_hline_y(runs, cpt_dataset, loss_margin):
    """Same horizontal loss line as the mean-OLMES thresholded plot: ``(1+margin)×min`` AdamW CPT loss."""
    _, adamw_ys, _ = _series_for_technique_olmes_avg(
        runs, "adamw", None, cpt_dataset, OLMES_EVAL_TASKS
    )
    if not adamw_ys:
        return None
    return float(min(adamw_ys)) * (1.0 + loss_margin)


def _olmes_task_group_label(task_name):
    for g, tasks in OLMES_EVAL_GROUPS.items():
        if task_name in tasks:
            return g
    return ""


def _sam_color_for_rho(rho, idx):
    if rho == 5e-2:
        return COLOR_MAP["sam"]
    if rho == 1e-1:
        return COLOR_MAP["sam_rho1e-1"]
    return plt.cm.Dark2(idx % 8)


def _print_skipped_cpt_lrs_olmes_loss(runs, cpt_dataset, cpt_tokens, midtrain_tokens, task_list, sam_rho, lr_range=None):
    """
    Print CPT LRs present in ``runs`` that are **not** included in ``_series_for_technique_olmes_avg``,
    with reasons (same filters as that helper).
    """
    lo, hi = _cpt_lr_window(cpt_dataset, lr_range)
    techs = [("adamw", None), ("sam", sam_rho)]
    print(
        f"[CPT LR diagnostics] {cpt_dataset} {cpt_tokens}M, {midtrain_tokens}B midtrain — "
        f"LR window [{lo:g}, {hi:g}] inclusive; OLMES must be complete; finetuning_val_loss required"
    )
    print(
        "  (If an LR appears for AdamW but not SAM, or vice versa, that LR exists only under that "
        "optimizer in the results — not a ‘skip’ for the other.)"
    )
    for optim, rho in techs:
        subset = [r for r in runs if r.get("optimizer") == optim and r.get("rho") == rho]
        subset = sorted(subset, key=lambda r: float(r.get("cpt_lr") or 0))
        in_data_lrs = [_lr_label(r.get("cpt_lr")) for r in subset if r.get("cpt_lr") is not None]
        skipped = []
        plotted_lrs = []
        for r in subset:
            lr_v = r.get("cpt_lr")
            oe = r.get("olmes_eval") or {}
            x_mean = _mean_olmes_tasks_if_complete(oe, task_list)
            loss = r.get("finetuning_val_loss")
            ok = (
                x_mean is not None
                and loss is not None
                and lr_v is not None
                and lo <= lr_v <= hi
            )
            if ok:
                plotted_lrs.append(_lr_label(lr_v))
                continue
            reasons = []
            if x_mean is None:
                missing = [t for t in task_list if t not in oe or oe[t] is None]
                if missing:
                    reasons.append(f"incomplete OLMES (missing/None: {missing})")
                else:
                    reasons.append("incomplete OLMES")
            if loss is None:
                reasons.append("finetuning_val_loss missing")
            if lr_v is None:
                reasons.append("cpt_lr missing")
            elif not (lo <= lr_v <= hi):
                reasons.append(f"cpt_lr outside [{lo:g}, {hi:g}]")
            lr_str = _lr_label(lr_v) if lr_v is not None else "None"
            skipped.append((lr_str, lr_v, "; ".join(reasons)))
        print(
            f"  {optim.upper()} — CPT LRs present in runs ({len(subset)}): "
            f"{', '.join(in_data_lrs) if in_data_lrs else '(none)'}"
        )
        print(f"  {optim.upper()} — plotted CPT LRs ({len(plotted_lrs)}): {', '.join(plotted_lrs)}")
        if skipped:
            print(f"  {optim.upper()} — skipped {len(skipped)} / {len(subset)} runs:")
            for lr_str, lr_v, why in skipped:
                print(f"    LR {lr_str} ({lr_v!r}): {why}")
        elif subset:
            print(f"  {optim.upper()} — none skipped (every run in window passes filters)")
        else:
            print(f"  {optim.upper()} — no CPT runs in this filter")

ARROW_BLACK = "black"


def _highest_x_intersection_on_polyline(x_poly, y_poly, y_h, eps=1e-12):
    """
    Among intersections of the horizontal ``y=y_h`` with the polyline, return the **maximum**
    ``x`` (highest average OLMES at that level).
    """
    best_x = None
    if not x_poly or len(x_poly) < 2:
        return None
    for i in range(len(x_poly) - 1):
        x0, y0 = float(x_poly[i]), float(y_poly[i])
        x1, y1 = float(x_poly[i + 1]), float(y_poly[i + 1])
        lo, hi = min(y0, y1), max(y0, y1)
        if y_h < lo - eps or y_h > hi + eps:
            continue
        if abs(y1 - y0) < 1e-15:
            if abs(y0 - y_h) < 1e-9:
                x_seg = max(x0, x1)
                best_x = x_seg if best_x is None else max(best_x, x_seg)
            continue
        t = (y_h - y0) / (y1 - y0)
        if -eps <= t <= 1.0 + eps:
            x = x0 + t * (x1 - x0)
            best_x = x if best_x is None else max(best_x, x)
    return best_x


def _hull_cut_highest_x_for_horizontal(x_hull, y_hull, y_line, eps=1e-12):
    """Highest-``x`` intersection of ``y=y_line`` with the hull polyline; clamp ``y`` into hull span if needed."""
    if not x_hull or len(x_hull) < 2:
        return None, None
    y_lo = min(float(y) for y in y_hull)
    y_hi = max(float(y) for y in y_hull)
    x = _highest_x_intersection_on_polyline(x_hull, y_hull, y_line, eps=eps)
    if x is not None:
        return x, y_line
    y_eff = min(max(y_line, y_lo), y_hi)
    x2 = _highest_x_intersection_on_polyline(x_hull, y_hull, y_eff, eps=eps)
    if x2 is not None:
        return x2, y_eff
    return None, None


def _olmes_pct_change_vs_baseline(baseline_x, x_at_hull):
    if baseline_x is None or x_at_hull is None:
        return None
    if abs(baseline_x) < 1e-15:
        return None
    return 100.0 * (x_at_hull - baseline_x) / baseline_x


def _format_pct_change(pct):
    if pct is None or not math.isfinite(pct):
        return None
    if pct <= 0:
        return f"{abs(pct):.1f}%"
    return f"{pct:.1f}%"


def _relative_olmes_drop_reduction_sam_vs_adamw(pct_adam, pct_sam):
    """``(|ΔAdamW| − |ΔSAM|) / |ΔAdamW|`` in percent when both are defined."""
    if pct_adam is None or pct_sam is None:
        return None
    if not math.isfinite(pct_adam) or not math.isfinite(pct_sam):
        return None
    mag_a = abs(float(pct_adam))
    mag_s = abs(float(pct_sam))
    if mag_a < 1e-12:
        return None
    reduction_pct = 100.0 * (mag_a - mag_s) / mag_a
    return mag_a, mag_s, reduction_pct


def _draw_sam_vs_adamw_improvement_box(ax, pct_adam, pct_sam):
    triple = _relative_olmes_drop_reduction_sam_vs_adamw(pct_adam, pct_sam)
    if triple is None:
        return
    _ma, _ms, red = triple
    body = f"SAM reduces forgetting by {red:.1f}%"
    ax.text(
        0.02,
        0.98,
        body,
        color="black",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONTSIZE["TICKS"] - 1,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.35", alpha=0.94),
        zorder=10,
    )


def _draw_black_arrow_baseline_to_hull(
    ax,
    label_name,
    baseline_x,
    x_hull,
    y_hull,
    y_line,
    zorder=6,
    verbose=True,
    arrow_lw=2.0,
    arrow_mutation_scale=14,
):
    """Arrow from baseline ``x`` to max-``x`` hull cut at ``y=y_line``; returns pct diagnostics."""
    if baseline_x is None or y_line is None:
        if verbose:
            print(f"{label_name}: skip (no baseline or horizontal line)")
        return None, None, None, None, None
    x_hit, y_draw = _hull_cut_highest_x_for_horizontal(x_hull, y_hull, y_line)
    if x_hit is None:
        if verbose:
            print(f"{label_name}: no hull cut (need ≥2 hull vertices)")
        return None, None, None, None, None
    pct = _olmes_pct_change_vs_baseline(baseline_x, x_hit)
    pct_str = _format_pct_change(pct)
    if verbose:
        if y_draw is not None and abs(y_draw - y_line) > 1e-9:
            print(
                f"{label_name}: hline y={y_line:.6g} outside hull y-range; "
                f"using y={y_draw:.6g}; max-x hull cut x={x_hit:.4f}"
            )
        else:
            print(
                f"{label_name}: max-x hull cut at x={x_hit:.4f} (y={y_draw:.6g}); "
                f"baseline x={baseline_x:.4f}; vs baseline: {pct_str if pct_str else 'n/a'}"
            )
    if abs(x_hit - baseline_x) < 1e-9:
        return x_hit, y_draw, pct_str, 0.5 * (baseline_x + x_hit), pct
    ax.annotate(
        "",
        xy=(x_hit, y_draw),
        xytext=(baseline_x, y_draw),
        arrowprops=dict(
            arrowstyle="->",
            color=ARROW_BLACK,
            lw=arrow_lw,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=arrow_mutation_scale,
        ),
        zorder=zorder,
    )
    mid_x = 0.5 * (baseline_x + x_hit)
    return x_hit, y_draw, pct_str, mid_x, pct


def _draw_stacked_sam_adamw_labels_below_arrows(ax, sam_pct_str, adamw_pct_str, y_arrow, x_center=None):
    if not sam_pct_str and not adamw_pct_str:
        return
    y_lo, y_hi = ax.get_ylim()
    y_span = y_hi - y_lo if y_hi != y_lo else 1.0
    y_anchor = y_arrow - 0.01 * y_span
    if x_center is None:
        x_center = float(np.mean(ax.get_xlim()))
    fs = FONTSIZE["TICKS"]
    dy_adamw_below_sam_pt = -(fs * 1.35 + 6.0)
    fig = ax.figure
    if sam_pct_str:
        ax.text(
            x_center,
            y_anchor,
            f"SAM: {sam_pct_str}",
            ha="center",
            va="top",
            fontsize=fs,
            color=ARROW_BLACK,
            zorder=8,
        )
    if adamw_pct_str:
        if sam_pct_str:
            trans = offset_copy(ax.transData, fig=fig, y=dy_adamw_below_sam_pt, units="points")
            ax.text(
                x_center,
                y_anchor,
                f"AdamW: {adamw_pct_str}",
                transform=trans,
                ha="center",
                va="top",
                fontsize=fs,
                color=ARROW_BLACK,
                zorder=8,
            )
        else:
            ax.text(
                x_center,
                y_anchor,
                f"AdamW: {adamw_pct_str}",
                ha="center",
                va="top",
                fontsize=fs,
                color=ARROW_BLACK,
                zorder=8,
            )


def _pick_sam_downstream_baseline_run(
    results,
    pretrain_token,
    midtrain_tokens,
    rho,
    prefer_eval_types=("hf", "olmo"),
):
    """Pre-CPT SAM downstream run at ``rho``; prefers hf over olmo (matches AdamW baseline logic)."""
    cands = [
        r
        for r in results
        if r.get("run_type") == "downstream"
        and r.get("optimizer") == "sam"
        and r.get("midtrain_tokens") == midtrain_tokens
        and (pretrain_token is None or r.get("pretrain_token") == pretrain_token)
        and r.get("rho") is not None
        and math.isclose(float(r.get("rho")), float(rho), rel_tol=0.0, abs_tol=1e-12)
    ]
    if not cands:
        return None
    for et in prefer_eval_types:
        for r in cands:
            if r.get("eval_type") == et:
                return r
    return cands[0]


def _sam_baseline_olmes_avg(results, pretrain_token, midtrain_tokens, task_list, rho):
    base = _pick_sam_downstream_baseline_run(results, pretrain_token, midtrain_tokens, rho)
    if base is None:
        return None
    oe = base.get("olmes_eval") or {}
    return _mean_olmes_tasks_if_complete(oe, task_list)


def plot_thresholded_pareto_olmes_raw(
    results,
    midtrain_tokens,
    cpt_dataset,
    cpt_tokens,
    out_path,
    pretrain_token=None,
    sam_rhos=(5e-2, 1e-1),
    loss_margin=0.002,
):
    """
    Single-panel Pareto: x = mean OLMES score, y = finetuning val loss.
    Uses **final** CPT evals only; scatter all runs, then ``_hull_xy_with_lr`` / ``_convex_hull_boundary``.
    Dashed vertical lines at AdamW and SAM base OLMES averages (not in legend). Horizontal line at
    ``(1 + loss_margin)`` times the **minimum** finetuning loss among AdamW CPT points (default
    ``loss_margin=0.005`` → 0.5% above best AdamW CPT loss). X-axis reversed (higher OLMES to the left).
    """
    runs = get_cpt_run_info(
        results,
        midtrain_tokens=midtrain_tokens,
        cpt_dataset=cpt_dataset,
        cpt_tokens=cpt_tokens,
        pretrain_token=pretrain_token,
        step=-1,
    )
    if not runs:
        print(f"No CPT runs for {cpt_dataset} {cpt_tokens}M, midtrain={midtrain_tokens}B (final eval)")
        return

    for sam_rho in sam_rhos:
        _print_skipped_cpt_lrs_olmes_loss(
            runs, cpt_dataset, cpt_tokens, midtrain_tokens, OLMES_EVAL_TASKS, sam_rho, lr_range=None
        )

    adamw_x = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, OLMES_EVAL_TASKS)
    sam_x_by_rho = {
        rho: _sam_baseline_olmes_avg(results, pretrain_token, midtrain_tokens, OLMES_EVAL_TASKS, rho)
        for rho in sam_rhos
    }

    loss_hline_y = _loss_threshold_hline_y(runs, cpt_dataset, loss_margin)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    techs = [("adamw", None)] + [("sam", rho) for rho in sam_rhos]
    colors_fixed = {
        ("adamw", None): COLOR_MAP["adamw"],
        ("sam", 5e-2): COLOR_MAP["sam"],
        ("sam", 1e-1): COLOR_MAP["sam_rho1e-1"],
    }
    markers = ["o", "s", "D"]

    lr_fs = max(FONTSIZE["TICKS"] - 4, 6)
    hull_by_optim = {}
    for i, (optim, rho) in enumerate(techs):
        x_vals, y_vals, lr_vals = _series_for_technique_olmes_avg(runs, optim, rho, cpt_dataset, OLMES_EVAL_TASKS)
        if not x_vals:
            continue
        color = colors_fixed[(optim, rho)]
        label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
        ax.scatter(x_vals, y_vals, marker=markers[i % len(markers)], color=color, label=label, alpha=ALPHA)
        if lr_vals is not None and len(lr_vals) == len(x_vals):
            for xv, yv, lr in zip(x_vals, y_vals, lr_vals):
                lab = _lr_label(lr)
                if lab:
                    ax.annotate(
                        lab,
                        (xv, yv),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=lr_fs,
                        color=color,
                        alpha=0.95,
                        zorder=5,
                    )
        x_hull, y_hull = _hull_xy_with_lr(x_vals, y_vals, lr_vals)
        hull_by_optim[(optim, rho)] = (x_hull, y_hull)
        if len(x_hull) >= 2:
            ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5, zorder=4)

    if adamw_x is not None:
        ax.axvline(adamw_x, color=COLOR_MAP["adamw"], linestyle="--", linewidth=2, alpha=0.75, zorder=2)
    for rho in sam_rhos:
        sam_x = sam_x_by_rho.get(rho)
        if sam_x is not None:
            ax.axvline(
                sam_x,
                color=colors_fixed[("sam", rho)],
                linestyle="--",
                linewidth=2,
                alpha=0.75,
                zorder=2,
            )
    if loss_hline_y is not None:
        ax.axhline(
            loss_hline_y,
            color="0.35",
            linestyle=":",
            linewidth=2,
            alpha=0.9,
            zorder=2,
        )

    ax.invert_xaxis()

    pct_adam_f = pct_sam_f = None
    if loss_hline_y is not None:
        print(f"Loss threshold line: y = {loss_hline_y:.6g} ((1+loss_margin)×min AdamW CPT loss)")
        xh_a, yh_a = hull_by_optim.get(("adamw", None), ([], []))
        xh_s, yh_s = hull_by_optim.get(("sam", 5e-2), ([], []))
        res_adam = _draw_black_arrow_baseline_to_hull(
            ax, "AdamW", adamw_x, xh_a, yh_a, loss_hline_y, verbose=True
        )
        res_sam = _draw_black_arrow_baseline_to_hull(
            ax,
            "SAM ρ=5e-2",
            sam_x_by_rho.get(5e-2),
            xh_s,
            yh_s,
            loss_hline_y,
            verbose=True,
        )
        y_d_adam, pct_adam, mid_adam, pct_adam_f = res_adam[1], res_adam[2], res_adam[3], res_adam[4]
        y_d_sam, pct_sam, mid_sam, pct_sam_f = res_sam[1], res_sam[2], res_sam[3], res_sam[4]
        y_arrow = y_d_adam if y_d_adam is not None else y_d_sam
        mids = [m for m in (mid_adam, mid_sam) if m is not None]
        x_label = float(np.mean(mids)) if mids else None
        if y_arrow is not None:
            _draw_stacked_sam_adamw_labels_below_arrows(ax, pct_sam, pct_adam, y_arrow, x_center=x_label)
        triple = _relative_olmes_drop_reduction_sam_vs_adamw(pct_adam_f, pct_sam_f)
        if triple is not None:
            _ma, _ms, red = triple
            print(f"SAM reduces forgetting by {red:.1f}%")

    if loss_hline_y is not None:
        _draw_sam_vs_adamw_improvement_box(ax, pct_adam_f, pct_sam_f)

    ax.set_xlabel("Average PT Eval Score", fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel("FT Validation Loss", fontsize=FONTSIZE["AXIS"])
    ax.set_title(
        f"SFT on {CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset)}",
        fontsize=FONTSIZE["TITLE"],
    )
    ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
    ax.grid(True, alpha=GRID_ALPHA)
    # Put horizontal legend at bottom
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.3),
            fancybox=True,
            shadow=False,
            ncol=3,
            fontsize=FONTSIZE["TICKS"] - 1,
            framealpha=0.92,
            handlelength=2.5,
            columnspacing=1.5,
        )

    # fig.suptitle(
    #     "Thresholded Pareto 1B — OLMES avg (raw) vs finetuning loss, CPT-LR convex hull (final CPT)",
    #     fontsize=FONTSIZE["TITLE"] + 2,
    #     y=1.02,
    # )
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")


def plot_thresholded_pareto_olmes_per_task_grid(
    results,
    midtrain_tokens,
    cpt_dataset,
    cpt_tokens,
    out_path,
    pretrain_token=None,
    sam_rhos=(5e-2,),
    loss_margin=0.005,
):
    """
    One subplot per OLMES eval task: x = that task's raw score, y = finetuning val loss.
    Uses the **same** horizontal loss threshold as the mean-OLMES figure
    (``(1+loss_margin)×min`` AdamW CPT loss from the **mean** OLMES series).
    AdamW/SAM vertical baselines use that task's downstream score only.
    """
    runs = get_cpt_run_info(
        results,
        midtrain_tokens=midtrain_tokens,
        cpt_dataset=cpt_dataset,
        cpt_tokens=cpt_tokens,
        pretrain_token=pretrain_token,
        step=-1,
    )
    if not runs:
        print(f"[per-task] No CPT runs for {cpt_dataset} {cpt_tokens}M, midtrain={midtrain_tokens}B")
        return

    loss_hline_y = _loss_threshold_hline_y(runs, cpt_dataset, loss_margin)
    techs = [("adamw", None)] + [("sam", rho) for rho in sam_rhos]
    colors_fixed = {("adamw", None): COLOR_MAP["adamw"]}
    for j, rho in enumerate(sam_rhos):
        colors_fixed[("sam", rho)] = _sam_color_for_rho(rho, j)
    markers = ["o", "s", "D", "v", "^", "P"]

    tasks = list(OLMES_EVAL_TASKS)
    ncols = 4
    nrows = int(math.ceil(len(tasks) / ncols))
    fig_w, fig_h = 3.15 * ncols, 2.45 * nrows
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True, squeeze=False)

    legend_handles = []
    legend_labels = []
    for ti, task_name in enumerate(tasks):
        ax = axs.flat[ti]
        row, col = ti // ncols, ti % ncols
        adamw_bx = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, [task_name])
        sam_x_by_rho = {
            rho: _sam_baseline_olmes_avg(results, pretrain_token, midtrain_tokens, [task_name], rho)
            for rho in sam_rhos
        }
        hull_by_optim = {}
        for i, (optim, rho) in enumerate(techs):
            x_vals, y_vals, lr_vals = _series_for_technique_olmes_single_task(
                runs, optim, rho, cpt_dataset, task_name
            )
            if not x_vals:
                continue
            color = colors_fixed[(optim, rho)]
            label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
            sc = ax.scatter(
                x_vals, y_vals, marker=markers[i % len(markers)], color=color, label=label, alpha=ALPHA
            )
            if not any(lbl == label for lbl in legend_labels):
                legend_handles.append(sc)
                legend_labels.append(label)
            x_hull, y_hull = _hull_xy_with_lr(x_vals, y_vals, lr_vals)
            hull_by_optim[(optim, rho)] = (x_hull, y_hull)
            if len(x_hull) >= 2:
                ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=1.6, zorder=4)

        if adamw_bx is not None:
            ax.axvline(
                adamw_bx, color=COLOR_MAP["adamw"], linestyle="--", linewidth=1.2, alpha=0.75, zorder=2
            )
        for rho in sam_rhos:
            sx = sam_x_by_rho.get(rho)
            if sx is not None:
                ax.axvline(
                    sx,
                    color=colors_fixed[("sam", rho)],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                    zorder=2,
                )
        if loss_hline_y is not None:
            ax.axhline(
                loss_hline_y, color="0.35", linestyle=":", linewidth=1.4, alpha=0.9, zorder=2
            )

        if loss_hline_y is not None:
            xh_a, yh_a = hull_by_optim.get(("adamw", None), ([], []))
            _draw_black_arrow_baseline_to_hull(
                ax,
                f"{task_name} AdamW",
                adamw_bx,
                xh_a,
                yh_a,
                loss_hline_y,
                verbose=False,
                arrow_lw=1.15,
                arrow_mutation_scale=9,
            )
            for rho in sam_rhos:
                xh_s, yh_s = hull_by_optim.get(("sam", rho), ([], []))
                _draw_black_arrow_baseline_to_hull(
                    ax,
                    f"{task_name} SAM",
                    sam_x_by_rho.get(rho),
                    xh_s,
                    yh_s,
                    loss_hline_y,
                    verbose=False,
                    arrow_lw=1.15,
                    arrow_mutation_scale=9,
                )

        ax.invert_xaxis()
        grp = _olmes_task_group_label(task_name)
        short = task_name.replace("::olmes", "")
        title = f"{grp}: {short}" if grp else short
        ax.set_title(title, fontsize=FONTSIZE["TICKS"] - 1)
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 2)
        ax.grid(True, alpha=GRID_ALPHA)
        if col == 0:
            ax.set_ylabel("FT val loss", fontsize=FONTSIZE["TICKS"] - 1)
        if row == nrows - 1:
            ax.set_xlabel("PT score (task)", fontsize=FONTSIZE["TICKS"] - 1)

    for j in range(len(tasks), nrows * ncols):
        axs.flat[j].set_visible(False)

    ds_title = CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset)
    y_txt = f"{loss_hline_y:.5g}" if loss_hline_y is not None else "n/a"
    fig.suptitle(
        f"{ds_title} — per-task OLMES vs FT loss; loss threshold y = {y_txt} "
        f"((1+{loss_margin})×min AdamW CPT loss, mean-OLMES series)",
        fontsize=FONTSIZE["TITLE"] - 1,
        y=1.01,
    )
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(legend_labels)),
            fontsize=FONTSIZE["TICKS"] - 2,
            framealpha=0.92,
        )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")


# 4T / 50B midtrain: OLMES avg vs finetuning loss (AdamW + SAM ρ=5e-2 + SAM ρ=1e-1). (cpt_dataset, cpt_tokens_M).
THRESHOLDED_OLMES_LOSS_4T_50B_50M = [
    ("tulu", 50),
    ("stackmathqa", 50),
    ("musicpile", 50),
    ("meta-math", 80),
    ("magicoder", 50),
]

# Relative margin above **min AdamW CPT loss** for the horizontal line: ``(1 + loss_margin) × min``.
LOSS_MARGIN_4T_50B_BY_DATASET = {
    "tulu": 0.005,
    "stackmathqa": 0.005,
    "musicpile": 0.005,
    "meta-math": 0.005,
    "magicoder": 0.005,
}


def main():
    results = parse_results()
    out_dir = os.path.join(RESULTS_DIR, "plots", "cpt_pareto", "thresholded_1b")
    os.makedirs(out_dir, exist_ok=True)
    for cpt_dataset, cpt_tokens in THRESHOLDED_OLMES_LOSS_4T_50B_50M:
        fname = f"{cpt_dataset}_pareto_1b.png"
        lm = 0.005
        plot_thresholded_pareto_olmes_raw(
            results,
            midtrain_tokens=50,
            cpt_dataset=cpt_dataset,
            cpt_tokens=cpt_tokens,
            out_path=os.path.join(out_dir, fname),
            pretrain_token=4,
            sam_rhos=(5e-2,),
            loss_margin=lm,
        )
        plot_thresholded_pareto_olmes_per_task_grid(
            results,
            midtrain_tokens=50,
            cpt_dataset=cpt_dataset,
            cpt_tokens=cpt_tokens,
            out_path=os.path.join(out_dir, f"{cpt_dataset}_pareto_1b_per_task.png"),
            pretrain_token=4,
            sam_rhos=(5e-2,),
            loss_margin=lm,
        )


if __name__ == "__main__":
    main()
