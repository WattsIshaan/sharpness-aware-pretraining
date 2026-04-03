"""
Pareto frontier plots for CPT: finetuning val loss (y) vs base eval task score (x).
One 2x3 figure per midtrain token budget (5B, 50B); each subplot = one base eval task.
One line per technique (AdamW, SAM with different rho); points = different cpt_lr.
X-axis only is reversed (higher score near origin).
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None

from utils.config_globals import RESULTS_DIR
from utils.plotting_globals import FONTSIZE, ALPHA, GRID_ALPHA, COLOR_MAP, CPT_DATASET_MAP
from preprocess_midtrain_olmo import (
    parse_results,
    get_cpt_run_info,
    filter_results,
    OLMO_TASK_NAMES,
    OLMES_EVAL_GROUPS,
    OLMES_EVAL_TASKS,
)

LR_RANGE = {
    "musicpile": (5e-5, 2e-4),
    "codealpaca": (1e-6, 5e-6),
    "tulu": (2e-6, 3e-5),
    "alpaca": (5e-6, 1e-4),
    "siqa": (3e-6, 1e-4),
    "gsm8k": (5e-6, 1e-4),
    "stackmathqa": (2e-5, 2e-4),
    "meta-math": (4e-5, 1e-4),
    "magicoder": (2e-5, 1.5e-4),
}


def _cpt_lr_window(cpt_dataset, lr_range_override=None):
    """
    Inclusive CPT LR bounds used to filter runs. If ``lr_range_override`` is ``(lo, hi)``, use it;
    otherwise ``LR_RANGE[cpt_dataset]``.
    """
    if lr_range_override is not None:
        return float(lr_range_override[0]), float(lr_range_override[1])
    lo, hi = LR_RANGE[cpt_dataset]
    return float(lo), float(hi)


# SAM rho values to include in the Pareto plot (None = plot all present in data)
CPT_PARETO_SAM_RHOS = [5e-2, 1e-1, 1.5e-1, 2e-1]

# Finetuning datasets for OLMES eval Pareto
OLMES_EVAL_DATASETS = [("tulu", 50), ("stackmathqa", 50), ("musicpile", 50)]

# Plot 1: effect of datasets — 6 datasets (order for 2x3)
# 20M: alpaca, codealpaca, gsm8k; 50M: tulu, stackmathqa, musicpile
EFFECT_OF_DATASETS_TASKS = [
    ("stackmathqa", 50),
    ("gsm8k", 20),
    ("tulu", 50),
    ("alpaca", 20),
    ("codealpaca", 20),
    ("musicpile", 50),
]
# Plot 2: effect of finetuning tokens — 2 rows (20M, 50M) x 3 cols (tulu, stackmathqa, musicpile)
EFFECT_OF_TOKENS_TASKS = [
    ("tulu", 20),
    ("stackmathqa", 20),
    ("musicpile", 20),
    ("tulu", 50),
    ("stackmathqa", 50),
    ("musicpile", 50),
]

# AdamW: contrasting color; SAM: gradient by rho (colormap name)
ADAMW_COLOR = "red"
SAM_CMAP = "viridis"


def _rho_label(rho):
    """Format rho for legend."""
    if rho is None:
        return ""
    s = f"{rho:.2e}".replace("+", "")
    c, e = s.split("e")
    c = c.rstrip("0").rstrip(".")
    return f"{c}e{e}"


def _lr_label(lr):
    """Format cpt_lr for annotation (compact)."""
    if lr is None:
        return ""
    s = f"{lr:.2e}".replace("+", "")
    c, e = s.split("e")
    c = c.rstrip("0").rstrip(".")
    return f"{c}e{e}"


def _collect_techniques(runs, sam_rhos=None):
    """
    From list of runs, get (optim, rho) to plot. AdamW always; SAM only for rhos in sam_rhos.
    If sam_rhos is None, include all SAM rhos present in data.
    """
    techs = set()
    for r in runs:
        opt = r.get("optimizer", "adamw")
        rho = r.get("rho")
        if opt == "adamw":
            techs.add((opt, rho))
        elif opt == "sam" and (sam_rhos is None or rho in sam_rhos):
            techs.add((opt, rho))
    adamw = [(o, p) for o, p in techs if o == "adamw"]
    sam = [(o, p) for o, p in techs if o == "sam"]
    sam.sort(key=lambda x: (x[1] or 0))
    return adamw + sam


def _relative_forgetting_pct(baseline, after):
    """Relative forgetting: (baseline - after) / baseline * 100."""
    if baseline is None or after is None or baseline == 0:
        return None
    return (baseline - after) / baseline * 100.0


def _transform_series_to_forgetting(x_vals, y_vals, lr_vals, baseline):
    """
    Convert x-series from score to relative forgetting (%), keeping y/lr alignment.
    Drops points where forgetting cannot be computed.
    """
    if not x_vals:
        return None, None, None
    x_new, y_new, lr_new = [], [], []
    for x, y, lr in zip(x_vals, y_vals, lr_vals):
        fx = _relative_forgetting_pct(baseline, x)
        if fx is None:
            continue
        x_new.append(fx)
        y_new.append(y)
        lr_new.append(lr)
    return (x_new, y_new, lr_new) if x_new else (None, None, None)


def _pick_downstream_baseline_run(results, pretrain_token, midtrain_tokens, prefer_eval_types=("hf", "olmo")):
    """
    Pick the AdamW downstream baseline run for a given pretrain/midtrain config.
    Prefers eval_type order in prefer_eval_types.
    """
    cands = [
        r for r in results
        if r.get("run_type") == "downstream"
        and r.get("optimizer") == "adamw"
        and r.get("midtrain_tokens") == midtrain_tokens
        and (pretrain_token is None or r.get("pretrain_token") == pretrain_token)
    ]
    if not cands:
        return None
    for et in prefer_eval_types:
        for r in cands:
            if r.get("eval_type") == et:
                return r
    return cands[0]


def _mean_olmes_tasks_if_complete(oe, task_list):
    """
    Mean of ``olmes_eval`` over ``task_list``. Returns ``None`` if any listed task is missing or
    ``None`` (no partial averages — incomplete rows are skipped at the call site).
    """
    if not oe or not task_list:
        return None
    vals = []
    for t in task_list:
        if t not in oe or oe[t] is None:
            return None
        vals.append(float(oe[t]))
    return float(np.mean(vals))


def _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, task_list):
    """Baseline score from pre-CPT/SFT AdamW downstream olmes_eval average over task_list."""
    base = _pick_downstream_baseline_run(results, pretrain_token, midtrain_tokens, prefer_eval_types=("hf", "olmo"))
    if base is None:
        return None
    oe = base.get("olmes_eval") or {}
    return _mean_olmes_tasks_if_complete(oe, task_list)


def _baseline_task_metrics_avg_score(results, pretrain_token, midtrain_tokens):
    """Baseline score from pre-CPT/SFT AdamW downstream task_metrics average over OLMO_TASK_NAMES."""
    base = _pick_downstream_baseline_run(results, pretrain_token, midtrain_tokens, prefer_eval_types=("olmo", "hf"))
    if base is None:
        return None
    tm = base.get("task_metrics") or {}
    vals = [tm[t] for t in OLMO_TASK_NAMES if t in tm and tm[t] is not None]
    return float(np.mean(vals)) if vals else None


def _convex_hull_boundary(x_vals, y_vals, lr_vals=None):
    """
    Return convex hull boundary as (x_list, y_list). Points are assumed sorted by lr ascending.
    If lr_vals is provided, do not connect the hull vertex with lowest lr to the one with highest lr
    (open path). Otherwise return closed hull.
    """
    points = np.column_stack([x_vals, y_vals])
    n = len(points)
    if n == 0:
        return [], []
    if n == 1:
        return list(x_vals), list(y_vals)
    if n == 2:
        return list(x_vals), list(y_vals)
    try:
        if ConvexHull is None:
            # Fallback when scipy not installed: return points in x-order
            order = np.argsort(np.array(x_vals))
            return [x_vals[i] for i in order], [y_vals[i] for i in order]
        hull = ConvexHull(points)
        vert_idx = hull.vertices  # indices into points, ccw order
        nv = len(vert_idx)
        if lr_vals is None or len(lr_vals) != n:
            hull_pts = points[vert_idx]
            return hull_pts[:, 0].tolist(), hull_pts[:, 1].tolist()
        # lr at each hull vertex (same index as in points)
        lr_at_vert = np.array([float(lr_vals[i]) for i in vert_idx])
        pos_min = int(np.argmin(lr_at_vert))
        pos_max = int(np.argmax(lr_at_vert))
        # Full hull minus only the edge between lowest-lr and highest-lr: traverse from pos_min to pos_max the long way (all vertices)
        step = 1 if (pos_min + 1) % nv != pos_max else -1
        order = []
        k = pos_min
        for _ in range(nv):
            order.append(k)
            k = (k + step) % nv
        hull_pts = points[vert_idx[np.array(order)]]
        return hull_pts[:, 0].tolist(), hull_pts[:, 1].tolist()
    except Exception:
        # Fallback: points in x-order so a boundary line is still drawn
        if n >= 2:
            order = np.argsort(np.array(x_vals))
            return [x_vals[i] for i in order], [y_vals[i] for i in order]
        return [], []


def _series_for_technique_avg(runs, optim, rho, cpt_dataset, lr_range=None):
    """Get (x_vals, y_vals, lr_vals) with x = average score over all tasks, sorted by cpt_lr ascending."""
    lo, hi = _cpt_lr_window(cpt_dataset, lr_range)
    subset = [r for r in runs if r.get("optimizer") == optim and r.get("rho") == rho]
    subset = sorted(subset, key=lambda r: float(r.get("cpt_lr") or 0))
    x, y, lrs = [], [], []
    for r in subset:
        tm = r.get("task_metrics") or {}
        vals = [tm[t] for t in OLMO_TASK_NAMES if t in tm and tm[t] is not None]
        if not vals or r.get("finetuning_val_loss") is None:
            continue
        if not (lo <= r.get("cpt_lr") <= hi):
            continue
        x.append(np.mean(vals))
        y.append(r["finetuning_val_loss"])
        lrs.append(r.get("cpt_lr"))
    return (x, y, lrs) if x else (None, None, None)


def _series_for_technique_olmes_avg(runs, optim, rho, cpt_dataset, task_list, lr_range=None):
    """Get (x_vals, y_vals, lr_vals) with x = average over task_list from olmes_eval, sorted by cpt_lr."""
    lo, hi = _cpt_lr_window(cpt_dataset, lr_range)
    subset = [r for r in runs if r.get("optimizer") == optim and r.get("rho") == rho]
    subset = sorted(subset, key=lambda r: float(r.get("cpt_lr") or 0))
    x, y, lrs = [], [], []
    for r in subset:
        oe = r.get("olmes_eval") or {}
        x_mean = _mean_olmes_tasks_if_complete(oe, task_list)
        if x_mean is None or r.get("finetuning_val_loss") is None:
            continue
        if not (lo <= r.get("cpt_lr") <= hi):
            continue
        x.append(x_mean)
        y.append(r["finetuning_val_loss"])
        lrs.append(r.get("cpt_lr"))
    return (x, y, lrs) if x else (None, None, None)


def _series_for_technique_olmes_avg_y_task(runs, optim, rho, cpt_dataset, task_list, y_task_key, lr_range=None):
    """
    Same as ``_series_for_technique_olmes_avg`` but **y** = ``olmes_eval[y_task_key]`` (e.g. gsm8k,
    codex_humaneval) instead of finetuning val loss. Does not require ``finetuning_val_loss``.
    """
    lo, hi = _cpt_lr_window(cpt_dataset, lr_range)
    subset = [r for r in runs if r.get("optimizer") == optim and r.get("rho") == rho]
    subset = sorted(subset, key=lambda r: float(r.get("cpt_lr") or 0))
    x, y, lrs = [], [], []
    for r in subset:
        oe = r.get("olmes_eval") or {}
        x_mean = _mean_olmes_tasks_if_complete(oe, task_list)
        if x_mean is None:
            continue
        if y_task_key not in oe or oe[y_task_key] is None:
            continue
        if not (lo <= r.get("cpt_lr") <= hi):
            continue
        x.append(x_mean)
        y.append(float(oe[y_task_key]))
        lrs.append(r.get("cpt_lr"))
    return (x, y, lrs) if x else (None, None, None)


def _series_for_technique_olmes_single_task(runs, optim, rho, cpt_dataset, task_name, lr_range=None):
    """Get (x_vals, y_vals, lr_vals) with x = olmes_eval[task_name], sorted by cpt_lr."""
    lo, hi = _cpt_lr_window(cpt_dataset, lr_range)
    subset = [r for r in runs if r.get("optimizer") == optim and r.get("rho") == rho]
    subset = sorted(subset, key=lambda r: float(r.get("cpt_lr") or 0))
    x, y, lrs = [], [], []
    for r in subset:
        oe = r.get("olmes_eval") or {}
        if task_name not in oe or oe[task_name] is None or r.get("finetuning_val_loss") is None:
            continue
        if not (lo <= r.get("cpt_lr") <= hi):
            continue
        x.append(oe[task_name])
        y.append(r["finetuning_val_loss"])
        lrs.append(r.get("cpt_lr"))
    return (x, y, lrs) if x else (None, None, None)


def _plot_pareto_olmes_eval_avg(results, midtrain_tokens, datasets, title_prefix, out_path, pretrain_token=None, all_sam_rhos=False, sam_rhos_list=None, step=None):
    """One subplot per dataset; x = avg OLMES eval score, y = finetuning val loss. all_sam_rhos or sam_rhos_list for SAM rhos. step: eval step (e.g. 4000) to filter CPT runs."""
    if sam_rhos_list is not None:
        techs = [("adamw", None)] + [("sam", r) for r in sam_rhos_list]
        colors_fixed = {("adamw", None): ADAMW_COLOR}
        for i, r in enumerate(sam_rhos_list):
            colors_fixed[("sam", r)] = [COLOR_MAP["sam"], "sienna"][i % 2]
        norm, cmap = None, None
    elif all_sam_rhos:
        techs = None
        colors_fixed = None
        norm = cmap = None
    else:
        techs = [("adamw", None), ("sam", 1e-1)]
        colors_fixed = {("adamw", None): ADAMW_COLOR, ("sam", 1e-1): COLOR_MAP["sam"]}
        norm = cmap = None
    markers = ["o", "s", "D"]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = np.ravel(axes)

    for idx, (cpt_dataset, cpt_tokens) in enumerate(datasets):
        ax = axes[idx]
        baseline_score = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, OLMES_EVAL_TASKS)
        runs = get_cpt_run_info(
            results,
            midtrain_tokens=midtrain_tokens,
            cpt_dataset=cpt_dataset,
            cpt_tokens=cpt_tokens,
            pretrain_token=pretrain_token,
            step=step,
        )
        if not runs:
            ax.set_title(CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset), fontsize=FONTSIZE["TITLE"])
            continue
        if all_sam_rhos:
            subplot_techs = _collect_techniques(runs, sam_rhos=None)
            sam_rhos_ordered = [r for o, r in subplot_techs if o == "sam" and r is not None]
            if sam_rhos_ordered:
                _norm = mcolors.LogNorm(vmin=min(sam_rhos_ordered), vmax=max(sam_rhos_ordered))
                _cmap = plt.get_cmap(SAM_CMAP)
            else:
                _norm = _cmap = None
            def _cf(o, r):
                if o == "adamw": return ADAMW_COLOR
                if _norm is None or r is None: return "C0"
                return _cmap(1.0 - _norm(r))
            subplot_techs = subplot_techs
        else:
            subplot_techs = techs
            _cf = lambda o, r: colors_fixed.get((o, r), "C0")
        for i, (optim, rho) in enumerate(subplot_techs):
            x_vals, y_vals, lr_vals = _series_for_technique_olmes_avg(
                runs, optim, rho, cpt_dataset, OLMES_EVAL_TASKS
            )
            x_vals, y_vals, lr_vals = _transform_series_to_forgetting(x_vals, y_vals, lr_vals, baseline_score)
            if not x_vals:
                continue
            label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
            color = _cf(optim, rho)
            ax.scatter(x_vals, y_vals, marker=markers[i % len(markers)], color=color, label=label, alpha=ALPHA)
            x_hull, y_hull = _convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
            if len(x_hull) >= 2:
                ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5,  zorder=4)
        ax.set_xlabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Finetuning val loss", fontsize=FONTSIZE["AXIS"])
        ax.set_title(CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset), fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.grid(True, alpha=GRID_ALPHA)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=FONTSIZE["LEGEND"] - 1,
            loc="upper center",
            ncol=min(len(labels), 5) if labels else 4,
            bbox_to_anchor=(0.5, 0.02),
        )
    fig.suptitle(
        f"{title_prefix} CPT Pareto for OLMES evals (avg)",
        fontsize=FONTSIZE["TITLE"] + 2,
        y=1.02,
    )
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_pareto_olmes_eval_single_task(
    results,
    midtrain_tokens,
    datasets,
    title_prefix,
    out_path,
    pretrain_token=None,
    all_sam_rhos=False,
    sam_rhos_list=None,
    task_name="gsm8k",
    step=None,
):
    """One subplot per dataset; x = OLMES eval score for task_name, y = finetuning val loss."""
    if sam_rhos_list is not None:
        techs = [("adamw", None)] + [("sam", r) for r in sam_rhos_list]
        colors_fixed = {("adamw", None): ADAMW_COLOR}
        for i, r in enumerate(sam_rhos_list):
            colors_fixed[("sam", r)] = [COLOR_MAP["sam"], "sienna"][i % 2]
        norm, cmap = None, None
    elif all_sam_rhos:
        techs = None
        colors_fixed = None
        norm = cmap = None
    else:
        techs = [("adamw", None), ("sam", 1e-1)]
        colors_fixed = {("adamw", None): ADAMW_COLOR, ("sam", 1e-1): COLOR_MAP["sam"]}
        norm = cmap = None

    markers = ["o", "s", "D"]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = np.ravel(axes)

    for idx, (cpt_dataset, cpt_tokens) in enumerate(datasets):
        ax = axes[idx]
        baseline_score = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, [task_name])
        runs = get_cpt_run_info(
            results,
            midtrain_tokens=midtrain_tokens,
            cpt_dataset=cpt_dataset,
            cpt_tokens=cpt_tokens,
            pretrain_token=pretrain_token,
            step=step,
        )
        if not runs:
            ax.set_title(CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset), fontsize=FONTSIZE["TITLE"])
            continue

        if all_sam_rhos:
            subplot_techs = _collect_techniques(runs, sam_rhos=None)
            sam_rhos_ordered = [r for o, r in subplot_techs if o == "sam" and r is not None]
            if sam_rhos_ordered:
                _norm = mcolors.LogNorm(vmin=min(sam_rhos_ordered), vmax=max(sam_rhos_ordered))
                _cmap = plt.get_cmap(SAM_CMAP)
            else:
                _norm = _cmap = None

            def _cf(o, r):
                if o == "adamw":
                    return ADAMW_COLOR
                if _norm is None or r is None:
                    return "C0"
                return _cmap(1.0 - _norm(r))

            subplot_techs = subplot_techs
        else:
            subplot_techs = techs
            _cf = lambda o, r: colors_fixed.get((o, r), "C0")

        for i, (optim, rho) in enumerate(subplot_techs):
            x_vals, y_vals, lr_vals = _series_for_technique_olmes_single_task(
                runs, optim, rho, cpt_dataset, task_name
            )
            x_vals, y_vals, lr_vals = _transform_series_to_forgetting(x_vals, y_vals, lr_vals, baseline_score)
            if not x_vals:
                continue
            label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
            color = _cf(optim, rho)
            ax.scatter(x_vals, y_vals, marker=markers[i % len(markers)], color=color, label=label, alpha=ALPHA)
            x_hull, y_hull = _convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
            if len(x_hull) >= 2:
                ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5, zorder=4)

        ax.set_xlabel(f"Relative forgetting on {task_name} (%)", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Finetuning val loss", fontsize=FONTSIZE["AXIS"])
        ax.set_title(CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset), fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.grid(True, alpha=GRID_ALPHA)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=FONTSIZE["LEGEND"] - 1,
            loc="upper center",
            ncol=min(len(labels), 5) if labels else 4,
            bbox_to_anchor=(0.5, 0.02),
        )
    fig.suptitle(
        f"{title_prefix} CPT Pareto for OLMES eval score ({task_name})",
        fontsize=FONTSIZE["TITLE"] + 2,
        y=1.02,
    )
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_pareto_task_metrics_avg(results, midtrain_tokens, datasets, title_prefix, out_path, pretrain_token=None, all_sam_rhos=False, sam_rhos_list=None, step=None):
    """Same layout as _plot_pareto_olmes_eval_avg but x = avg over OLMo downstream task_metrics (winogrande, mmlu, ...). step: filter CPT runs by eval step."""
    if sam_rhos_list is not None:
        techs = [("adamw", None)] + [("sam", r) for r in sam_rhos_list]
        colors_fixed = {("adamw", None): ADAMW_COLOR}
        for i, r in enumerate(sam_rhos_list):
            colors_fixed[("sam", r)] = [COLOR_MAP["sam"], "sienna"][i % 2]
        norm, cmap = None, None
    elif all_sam_rhos:
        techs = None
        colors_fixed = None
        norm = cmap = None
    else:
        techs = [("adamw", None), ("sam", 1e-1)]
        colors_fixed = {("adamw", None): ADAMW_COLOR, ("sam", 1e-1): COLOR_MAP["sam"]}
        norm = cmap = None
    markers = ["o", "s", "D"]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = np.ravel(axes)

    for idx, (cpt_dataset, cpt_tokens) in enumerate(datasets):
        ax = axes[idx]
        baseline_score = _baseline_task_metrics_avg_score(results, pretrain_token, midtrain_tokens)
        runs = get_cpt_run_info(
            results,
            midtrain_tokens=midtrain_tokens,
            cpt_dataset=cpt_dataset,
            cpt_tokens=cpt_tokens,
            pretrain_token=pretrain_token,
            step=step,
        )
        if not runs:
            ax.set_title(CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset), fontsize=FONTSIZE["TITLE"])
            continue
        if all_sam_rhos:
            subplot_techs = _collect_techniques(runs, sam_rhos=None)
            sam_rhos_ordered = [r for o, r in subplot_techs if o == "sam" and r is not None]
            if sam_rhos_ordered:
                _norm = mcolors.LogNorm(vmin=min(sam_rhos_ordered), vmax=max(sam_rhos_ordered))
                _cmap = plt.get_cmap(SAM_CMAP)
            else:
                _norm = _cmap = None
            def _cf(o, r):
                if o == "adamw": return ADAMW_COLOR
                if _norm is None or r is None: return "C0"
                return _cmap(1.0 - _norm(r))
            subplot_techs = subplot_techs
        else:
            subplot_techs = techs
            _cf = lambda o, r: colors_fixed.get((o, r), "C0")
        for i, (optim, rho) in enumerate(subplot_techs):
            x_vals, y_vals, lr_vals = _series_for_technique_avg(runs, optim, rho, cpt_dataset)
            x_vals, y_vals, lr_vals = _transform_series_to_forgetting(x_vals, y_vals, lr_vals, baseline_score)
            if not x_vals:
                continue
            label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
            color = _cf(optim, rho)
            ax.scatter(x_vals, y_vals, marker=markers[i % len(markers)], color=color, label=label, alpha=ALPHA)
            x_hull, y_hull = _convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
            if len(x_hull) >= 2:
                ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5,  zorder=4)
        ax.set_xlabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Finetuning val loss", fontsize=FONTSIZE["AXIS"])
        ax.set_title(CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset), fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.grid(True, alpha=GRID_ALPHA)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=FONTSIZE["LEGEND"] - 1,
            loc="upper center",
            ncol=min(len(labels), 5) if labels else 4,
            bbox_to_anchor=(0.5, 0.02),
        )
    fig.suptitle(
        f"{title_prefix} CPT Pareto for OLMo downstream task metrics (avg)",
        fontsize=FONTSIZE["TITLE"] + 2,
        y=1.02,
    )
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_pareto_olmes_eval_by_group(results, midtrain_tokens, datasets, title_prefix, out_path, pretrain_token=None, all_sam_rhos=False, sam_rhos_list=None):
    """Rows = finetuning datasets, cols = groups (mcq, generative, heldout). all_sam_rhos or sam_rhos_list for SAM rhos."""
    if sam_rhos_list is not None:
        techs = [("adamw", None)] + [("sam", r) for r in sam_rhos_list]
        colors_fixed = {("adamw", None): ADAMW_COLOR}
        for i, r in enumerate(sam_rhos_list):
            colors_fixed[("sam", r)] = [COLOR_MAP["sam"], "sienna"][i % 2]
        norm = cmap = None
    elif all_sam_rhos:
        techs = None
        colors_fixed = None
        norm = cmap = None
    else:
        techs = [("adamw", None), ("sam", 1e-1)]
        colors_fixed = {("adamw", None): ADAMW_COLOR, ("sam", 1e-1): COLOR_MAP["sam"]}
        norm = cmap = None
    markers = ["o", "s", "D"]
    groups = list(OLMES_EVAL_GROUPS.keys())
    nrows, ncols = len(datasets), len(groups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    legend_techs = techs

    for row, (cpt_dataset, cpt_tokens) in enumerate(datasets):
        baseline_by_group = {
            group_name: _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, task_list)
            for group_name, task_list in OLMES_EVAL_GROUPS.items()
        }
        runs = get_cpt_run_info(
            results,
            midtrain_tokens=midtrain_tokens,
            cpt_dataset=cpt_dataset,
            cpt_tokens=cpt_tokens,
            pretrain_token=pretrain_token,
        )
        if all_sam_rhos and runs:
            subplot_techs = _collect_techniques(runs, sam_rhos=None)
            if legend_techs is None:
                legend_techs = subplot_techs
            sam_rhos_ordered = [r for o, r in subplot_techs if o == "sam" and r is not None]
            if sam_rhos_ordered:
                _norm = mcolors.LogNorm(vmin=min(sam_rhos_ordered), vmax=max(sam_rhos_ordered))
                _cmap = plt.get_cmap(SAM_CMAP)
            else:
                _norm = _cmap = None
            def _cf(o, r):
                if o == "adamw": return ADAMW_COLOR
                if _norm is None or r is None: return "C0"
                return _cmap(1.0 - _norm(r))
        else:
            subplot_techs = techs
            _cf = lambda o, r: colors_fixed.get((o, r), "C0") if colors_fixed else "C0"
        for col, group_name in enumerate(groups):
            ax = axes[row, col]
            if not runs:
                ax.set_title(f"{CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset)}\n{group_name}", fontsize=FONTSIZE["TITLE"])
                continue
            task_list = OLMES_EVAL_GROUPS[group_name]
            for i, (optim, rho) in enumerate(subplot_techs):
                x_vals, y_vals, lr_vals = _series_for_technique_olmes_avg(
                    runs, optim, rho, cpt_dataset, task_list
                )
                x_vals, y_vals, lr_vals = _transform_series_to_forgetting(
                    x_vals, y_vals, lr_vals, baseline_by_group.get(group_name)
                )
                if not x_vals:
                    continue
                label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
                color = _cf(optim, rho)
                ax.scatter(
                    x_vals,
                    y_vals,
                    marker=markers[i % len(markers)],
                    color=color,
                    label=label,
                    alpha=ALPHA,
                )
                x_hull, y_hull = _convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
                if len(x_hull) >= 2:
                    ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5,  zorder=4)
            ax.set_xlabel(f"Relative forgetting ({group_name}) (%)", fontsize=FONTSIZE["AXIS"])
            ax.set_ylabel("Finetuning val loss", fontsize=FONTSIZE["AXIS"])
            dataset_label = CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset)
            ax.set_title(f"{dataset_label} — {group_name}", fontsize=FONTSIZE["TITLE"])
            ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
            ax.grid(True, alpha=GRID_ALPHA)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol_leg = len(legend_techs) if legend_techs is not None else (len(labels) if labels else 4)
    fig.legend(
        handles,
        labels,
        fontsize=FONTSIZE["LEGEND"] - 1,
        loc="upper center",
        ncol=ncol_leg,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(
        f"{title_prefix} CPT Pareto for OLMES evals by group",
        fontsize=FONTSIZE["TITLE"] + 2,
        y=1.02,
    )
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_pareto_olmes_4T_5B_50B_1x2(results, out_path, cpt_dataset="tulu", cpt_tokens=50, pretrain_token=4):
    """1x2 combined: left 4T 5B, right 4T 50B Tulu 50M OLMES avg Pareto. Share x and y."""
    SAM_RHO_50B = 5e-2
    markers = ["o", "s", "D"]
    # Precompute 5e-2 color from 5B's colormap so 50B matches 5B
    runs_5b = get_cpt_run_info(
        results, midtrain_tokens=5, cpt_dataset=cpt_dataset, cpt_tokens=cpt_tokens, pretrain_token=pretrain_token
    )
    color_5e2 = COLOR_MAP["sam"]
    if runs_5b:
        techs_5b = _collect_techniques(runs_5b, sam_rhos=None)
        sam_rhos_5b = [r for o, r in techs_5b if o == "sam" and r is not None]
        if sam_rhos_5b:
            _norm = mcolors.LogNorm(vmin=min(sam_rhos_5b), vmax=max(sam_rhos_5b))
            _cmap = plt.get_cmap(SAM_CMAP)
            try:
                color_5e2 = _cmap(1.0 - _norm(SAM_RHO_50B))
            except Exception:
                pass
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    configs = [
        (5, True, None),   # 5B, all_sam_rhos
        (50, False, [SAM_RHO_50B]),  # 50B, sam_rhos_list
    ]
    for idx, (midtrain_tokens, all_sam_rhos, sam_rhos_list) in enumerate(configs):
        ax = axes[idx]
        baseline_score = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, OLMES_EVAL_TASKS)
        runs = get_cpt_run_info(
            results,
            midtrain_tokens=midtrain_tokens,
            cpt_dataset=cpt_dataset,
            cpt_tokens=cpt_tokens,
            pretrain_token=pretrain_token,
        )
        if not runs:
            ax.set_title(f"4T {midtrain_tokens}B", fontsize=FONTSIZE["TITLE"])
            continue
        if all_sam_rhos:
            subplot_techs = _collect_techniques(runs, sam_rhos=None)
            sam_rhos_ordered = [r for o, r in subplot_techs if o == "sam" and r is not None]
            if sam_rhos_ordered:
                _norm = mcolors.LogNorm(vmin=min(sam_rhos_ordered), vmax=max(sam_rhos_ordered))
                _cmap = plt.get_cmap(SAM_CMAP)
            else:
                _norm = _cmap = None
            def _cf(o, r):
                if o == "adamw": return ADAMW_COLOR
                if o == "sam" and r == SAM_RHO_50B: return color_5e2
                if _norm is None or r is None: return "C0"
                return _cmap(1.0 - _norm(r))
        else:
            subplot_techs = [("adamw", None)] + [("sam", r) for r in sam_rhos_list]
            colors_fixed = {("adamw", None): ADAMW_COLOR, ("sam", SAM_RHO_50B): color_5e2}
            _cf = lambda o, r: colors_fixed.get((o, r), "C0")
        for i, (optim, rho) in enumerate(subplot_techs):
            x_vals, y_vals, lr_vals = _series_for_technique_olmes_avg(
                runs, optim, rho, cpt_dataset, OLMES_EVAL_TASKS
            )
            x_vals, y_vals, lr_vals = _transform_series_to_forgetting(x_vals, y_vals, lr_vals, baseline_score)
            if not x_vals:
                continue
            label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
            color = _cf(optim, rho)
            ax.scatter(x_vals, y_vals, marker=markers[i % len(markers)], color=color, label=label, alpha=ALPHA)
            x_hull, y_hull = _convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
            if len(x_hull) >= 2:
                ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5,  zorder=4)
        ax.set_xlabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Finetuning val loss", fontsize=FONTSIZE["AXIS"])
        ax.set_title(f"4T {midtrain_tokens}B", fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.grid(True, alpha=GRID_ALPHA)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    if handles:
        fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"] - 1, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("4T CPT OLMES Pareto (Tulu 50M)", fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# Effect of pretraining tokens: 2T vs 4T
PRETRAIN_1X2_DATASET = ("tulu", 50)  # single dataset for 1x2
PRETRAIN_2X3_DATASETS = [("tulu", 50), ("stackmathqa", 50), ("musicpile", 50)]  # 3 cols for 2x3
PRETRAIN_TOKENS = [2, 4]  # 2T, 4T


def _plot_effect_of_pretrain_1x2(results, out_path, midtrain_tokens=5, all_sam_rhos=True):
    """1x2: 2T vs 4T pretrain, same dataset (Tulu 50M), OLMES avg Pareto. Share x and y."""
    cpt_dataset, cpt_tokens = PRETRAIN_1X2_DATASET
    techs = None
    colors_fixed = None
    markers = ["o", "s", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for idx, pretrain_token in enumerate(PRETRAIN_TOKENS):
        ax = axes[idx]
        baseline_score = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, OLMES_EVAL_TASKS)
        runs = get_cpt_run_info(
            results,
            midtrain_tokens=midtrain_tokens,
            cpt_dataset=cpt_dataset,
            cpt_tokens=cpt_tokens,
            pretrain_token=pretrain_token,
        )
        if not runs:
            ax.set_title(f"{pretrain_token}T 5B", fontsize=FONTSIZE["TITLE"])
            continue
        if all_sam_rhos:
            subplot_techs = _collect_techniques(runs, sam_rhos=None)
            sam_rhos_ordered = [r for o, r in subplot_techs if o == "sam" and r is not None]
            if sam_rhos_ordered:
                _norm = mcolors.LogNorm(vmin=min(sam_rhos_ordered), vmax=max(sam_rhos_ordered))
                _cmap = plt.get_cmap(SAM_CMAP)
            else:
                _norm = _cmap = None
            def _cf(o, r):
                if o == "adamw": return ADAMW_COLOR
                if _norm is None or r is None: return "C0"
                return _cmap(1.0 - _norm(r))
        else:
            subplot_techs = [("adamw", None), ("sam", 1e-1)]
            _cf = lambda o, r: ADAMW_COLOR if o == "adamw" else COLOR_MAP["sam"]
        for i, (optim, rho) in enumerate(subplot_techs):
            x_vals, y_vals, lr_vals = _series_for_technique_olmes_avg(
                runs, optim, rho, cpt_dataset, OLMES_EVAL_TASKS
            )
            x_vals, y_vals, lr_vals = _transform_series_to_forgetting(x_vals, y_vals, lr_vals, baseline_score)
            if not x_vals:
                continue
            label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
            color = _cf(optim, rho)
            ax.scatter(x_vals, y_vals, marker=markers[i % len(markers)], color=color, label=label, alpha=ALPHA)
            x_hull, y_hull = _convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
            if len(x_hull) >= 2:
                ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5,  zorder=4)
        ax.set_xlabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Finetuning val loss", fontsize=FONTSIZE["AXIS"])
        ax.set_title(f"{pretrain_token}T {midtrain_tokens}B", fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.grid(True, alpha=GRID_ALPHA)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    if handles:
        fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"] - 1, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Effect of pretraining tokens (OLMES avg, Tulu 50M)", fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_effect_of_pretrain_2x3(results, out_path, midtrain_tokens=5, all_sam_rhos=True):
    """2x3: rows = 2T, 4T; cols = tulu 50M, stackmathqa 50M, musicpile 50M. OLMES avg Pareto. Share x and y across columns."""
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex="col", sharey="col")
    markers = ["o", "s", "D"]
    legend_handles, legend_labels = None, None
    for row, pretrain_token in enumerate(PRETRAIN_TOKENS):
        for col, (cpt_dataset, cpt_tokens) in enumerate(PRETRAIN_2X3_DATASETS):
            ax = axes[row, col]
            baseline_score = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, OLMES_EVAL_TASKS)
            runs = get_cpt_run_info(
                results,
                midtrain_tokens=midtrain_tokens,
                cpt_dataset=cpt_dataset,
                cpt_tokens=cpt_tokens,
                pretrain_token=pretrain_token,
            )
            if not runs:
                ax.set_title(f"{pretrain_token}T — {CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset)}", fontsize=FONTSIZE["TITLE"])
                continue
            if all_sam_rhos:
                subplot_techs = _collect_techniques(runs, sam_rhos=None)
                sam_rhos_ordered = [r for o, r in subplot_techs if o == "sam" and r is not None]
                if sam_rhos_ordered:
                    _norm = mcolors.LogNorm(vmin=min(sam_rhos_ordered), vmax=max(sam_rhos_ordered))
                    _cmap = plt.get_cmap(SAM_CMAP)
                else:
                    _norm = _cmap = None
                def _cf(o, r):
                    if o == "adamw": return ADAMW_COLOR
                    if _norm is None or r is None: return "C0"
                    return _cmap(1.0 - _norm(r))
            else:
                subplot_techs = [("adamw", None), ("sam", 1e-1)]
                _cf = lambda o, r: ADAMW_COLOR if o == "adamw" else COLOR_MAP["sam"]
            for i, (optim, rho) in enumerate(subplot_techs):
                x_vals, y_vals, lr_vals = _series_for_technique_olmes_avg(
                    runs, optim, rho, cpt_dataset, OLMES_EVAL_TASKS
                )
                x_vals, y_vals, lr_vals = _transform_series_to_forgetting(x_vals, y_vals, lr_vals, baseline_score)
                if not x_vals:
                    continue
                label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
                color = _cf(optim, rho)
                ax.scatter(x_vals, y_vals, marker=markers[i % len(markers)], color=color, label=label, alpha=ALPHA)
                x_hull, y_hull = _convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
                if len(x_hull) >= 2:
                    ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5,  zorder=4)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.set_xlabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
            ax.set_ylabel("Finetuning val loss", fontsize=FONTSIZE["AXIS"])
            ax.set_title(f"{pretrain_token}T — {CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset)}", fontsize=FONTSIZE["TITLE"])
            ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
            ax.grid(True, alpha=GRID_ALPHA)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, fontsize=FONTSIZE["LEGEND"] - 1, loc="upper center", ncol=min(len(legend_labels), 5), bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Effect of pretraining tokens (OLMES avg)", fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


PARETO_2X3_TASKS = [("stackmathqa", 50), ("tulu", 50), ("musicpile", 50)]


def _plot_pareto_2x3_avg(results, midtrain_tokens, out_path, tasks=None, show_tokens_in_title=False, share_axes_col=False, share_x_all=False, sam_rho=1e-1, sam_rhos=None, all_sam_rhos=False, pretrain_token=None):
    """
    2x3 subplot: each subplot = one CPT dataset, x = avg score over base eval tasks, y = finetuning val loss.
    If all_sam_rhos is True, include all SAM rho values with gradient coloring (AdamW red).
    Otherwise sam_rho or sam_rhos control which SAM rhos to plot.
    share_axes_col: share x and y within each column. share_x_all: share x-axis across all subplots.
    """
    if tasks is None:
        tasks = PARETO_2X3_TASKS
    colors_fixed = None
    legend_techs = techs = None
    if all_sam_rhos:
        techs = None  # will be set per-subplot from runs
    elif sam_rhos is not None and isinstance(sam_rhos, (list, tuple)):
        techs = [("adamw", None)] + [("sam", r) for r in sam_rhos]
        colors_fixed = {("adamw", None): ADAMW_COLOR}
        sam_colors = [COLOR_MAP["sam"], "sienna"]
        for i, r in enumerate(sam_rhos):
            colors_fixed[("sam", r)] = sam_colors[i % len(sam_colors)]
    else:
        techs = [("adamw", None), ("sam", sam_rho)]
        colors_fixed = {("adamw", None): ADAMW_COLOR, ("sam", sam_rho): COLOR_MAP["sam"]}
    markers = ["o", "s", "D"]

    n = len(tasks)
    nrows = 1 if n <= 3 else 2
    ncols = 3
    if share_x_all:
        share_kw = {"sharex": True, "sharey": False}
    elif share_axes_col:
        share_kw = {"sharex": "col", "sharey": "col"}
    else:
        share_kw = {}
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), **share_kw)
    axes = np.ravel(axes)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    for idx, (cpt_dataset, cpt_tokens) in enumerate(tasks):
        ax = axes[idx]
        baseline_score = _baseline_task_metrics_avg_score(results, pretrain_token, midtrain_tokens)
        runs = get_cpt_run_info(
            results,
            midtrain_tokens=midtrain_tokens,
            cpt_dataset=cpt_dataset,
            cpt_tokens=cpt_tokens,
            pretrain_token=pretrain_token,
        )
        if not runs:
            base_title = CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset)
            title = f"{base_title} {cpt_tokens}M" if show_tokens_in_title else base_title
            ax.set_title(title, fontsize=FONTSIZE["TITLE"])
            continue

        if all_sam_rhos:
            subplot_techs = _collect_techniques(runs, sam_rhos=None)
            if idx == 0:
                legend_techs = subplot_techs
            sam_rhos_ordered = [r for o, r in subplot_techs if o == "sam" and r is not None]
            if sam_rhos_ordered:
                _rho_min, _rho_max = min(sam_rhos_ordered), max(sam_rhos_ordered)
                _norm = mcolors.LogNorm(vmin=_rho_min, vmax=_rho_max)
                _cmap = plt.get_cmap(SAM_CMAP)
            else:
                _norm = None
                _cmap = None

            def _color_for(optim, rho):
                if optim == "adamw":
                    return ADAMW_COLOR
                if _norm is None or rho is None:
                    return "C0"
                return _cmap(1.0 - _norm(rho))

            subplot_techs_for_legend = subplot_techs
        else:
            subplot_techs = techs
            legend_techs = techs
            _color_for = lambda optim, rho: colors_fixed.get((optim, rho), "C0")
            subplot_techs_for_legend = techs

        for i, (optim, rho) in enumerate(subplot_techs):
            x_vals, y_vals, lr_vals = _series_for_technique_avg(runs, optim, rho, cpt_dataset)
            x_vals, y_vals, lr_vals = _transform_series_to_forgetting(x_vals, y_vals, lr_vals, baseline_score)
            if not x_vals:
                continue
            label = "AdamW" if optim == "adamw" else f"SAM ρ={_rho_label(rho)}"
            color = _color_for(optim, rho)
            ax.scatter(
                x_vals,
                y_vals,
                marker=markers[i % len(markers)],
                color=color,
                label=label,
                alpha=ALPHA,
                linewidth=2,
            )
            x_hull, y_hull = _convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
            if len(x_hull) >= 2:
                ax.plot(
                    x_hull,
                    y_hull,
                    color=color,
                    alpha=0.8,
                    linewidth=1.5,
                    zorder=0,
                )

        if idx % ncols == 0:
            ax.set_ylabel("Finetuning Loss", fontsize=FONTSIZE["AXIS"])
        if nrows > 1 and idx // ncols == nrows - 1:
            ax.set_xlabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
        elif nrows == 1:
            ax.set_xlabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
        base_title = CPT_DATASET_MAP.get(cpt_dataset, cpt_dataset)
        title = f"SFT on {base_title} {cpt_tokens}M" if show_tokens_in_title else f"SFT on {base_title}"
        ax.set_title(title, fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.grid(True, alpha=GRID_ALPHA)

    handles, labels = axes[0].get_legend_handles_labels()
    ncol_legend = min(len(legend_techs), 4) if legend_techs else 4
    fig.legend(
        handles,
        labels,
        fontsize=FONTSIZE["LEGEND"] - 1,
        loc="upper center",
        ncol=ncol_legend,
        bbox_to_anchor=(0.5, 0.02),
    )
    # fig.suptitle(
    #     "SFT-PT Pareto: (Finetuning Loss on Different Datasets v/s Average Base Eval Score)",
    #     fontsize=FONTSIZE["TITLE"] + 2,
    #     y=1.02,
    # )
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


SFT_LRS = [1e-5, 3e-5, 6e-5]


def _sft_score_at_lr(runs, lr, task_list):
    """Avg OLMES score over task_list for runs with this sft_lr. Returns None if no data."""
    subset = [r for r in runs if r.get("sft_lr") == lr]
    if not subset:
        return None
    vals = []
    for r in subset:
        oe = r.get("olmes_eval") or {}
        m = _mean_olmes_tasks_if_complete(oe, task_list)
        if m is not None:
            vals.append(m)
    return np.mean(vals) if vals else None


def _plot_sft_eval_avg(results, out_path, pretrain_token=4, midtrain_tokens=5):
    """Avg OLMES eval score vs increasing LR: AdamW (one line) + SAM (one line per rho)."""
    sft_runs = filter_results(
        results,
        finetune_type="SFT",
        pretrain_token=pretrain_token,
        midtrain_token=midtrain_tokens,
    )
    sft_runs = [r for r in sft_runs if r.get("sft_lr") in SFT_LRS]
    adamw_runs = [r for r in sft_runs if r.get("optimizer") == "adamw"]
    sam_runs = [r for r in sft_runs if r.get("optimizer") == "sam"]
    sam_rhos = sorted(set(r.get("rho") for r in sam_runs if r.get("rho") is not None))

    lrs = sorted(SFT_LRS)
    lr_labels = [_lr_label(lr) for lr in lrs]
    baseline_score = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, OLMES_EVAL_TASKS)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # AdamW
    adamw_scores = [
        _relative_forgetting_pct(baseline_score, _sft_score_at_lr(adamw_runs, lr, OLMES_EVAL_TASKS))
        for lr in lrs
    ]
    valid_a = [(x, y) for x, y in zip(lr_labels, adamw_scores) if y is not None]
    if valid_a:
        # Use numeric LR on x-axis so hull lines align correctly.
        x_num = [lr for lr, y in zip(lrs, adamw_scores) if y is not None]
        y_num = [y for y in adamw_scores if y is not None]
        ax.plot(x_num, y_num, marker="o", color=ADAMW_COLOR, label="AdamW", linewidth=2, alpha=ALPHA)
        x_hull, y_hull = _convex_hull_boundary(x_num, y_num, lr_vals=x_num)
        if len(x_hull) >= 2:
            ax.plot(x_hull, y_hull, color=ADAMW_COLOR, alpha=0.8, linewidth=1.5, zorder=0)
    # SAM: one line per rho
    if sam_rhos:
        if len(sam_rhos) >= 2:
            cmap = plt.get_cmap(SAM_CMAP)
            norm = mcolors.LogNorm(vmin=min(sam_rhos), vmax=max(sam_rhos))
        for rho in sam_rhos:
            subset = [r for r in sam_runs if r.get("rho") == rho]
            scores = [
                _relative_forgetting_pct(baseline_score, _sft_score_at_lr(subset, lr, OLMES_EVAL_TASKS))
                for lr in lrs
            ]
            valid = [(x, y) for x, y in zip(lr_labels, scores) if y is not None]
            if valid:
                x_num = [lr for lr, y in zip(lrs, scores) if y is not None]
                y_num = [y for y in scores if y is not None]
                color = cmap(1.0 - norm(rho)) if len(sam_rhos) >= 2 else COLOR_MAP["sam"]
                ax.plot(x_num, y_num, marker="s", color=color, label=f"SAM ρ={_rho_label(rho)}", linewidth=2, alpha=ALPHA)
                x_hull, y_hull = _convex_hull_boundary(x_num, y_num, lr_vals=x_num)
                if len(x_hull) >= 2:
                    ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5,  zorder=4)
    ax.set_xlabel("SFT learning rate", fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
    ax.set_title(f"SFT eval (4T, {midtrain_tokens}B)", fontsize=FONTSIZE["TITLE"])
    ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
    ax.set_xticks(lrs)
    ax.set_xticklabels(lr_labels, rotation=45)
    ax.grid(True, alpha=GRID_ALPHA)
    handles, labels = ax.get_legend_handles_labels()
    ncol_leg = min(len(labels), 5) if labels else 4
    fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"] - 1, loc="upper center", ncol=ncol_leg, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_sft_eval_4T_5B_50B_1x2(results, out_path, pretrain_token=4):
    """1x2 combined: left 4T 5B, right 4T 50B SFT avg OLMES vs LR. Share x and y."""
    SAM_RHO_50B = 5e-2
    lrs = sorted(SFT_LRS)
    lr_labels = [_lr_label(lr) for lr in lrs]
    # Precompute 5e-2 color from 5B's colormap so 50B matches 5B
    sft_5b = filter_results(results, finetune_type="SFT", pretrain_token=pretrain_token, midtrain_token=5)
    sft_5b = [r for r in sft_5b if r.get("sft_lr") in SFT_LRS]
    sam_rhos_5b = sorted(set(r.get("rho") for r in sft_5b if r.get("optimizer") == "sam" and r.get("rho") is not None))
    color_5e2 = COLOR_MAP["sam"]
    if len(sam_rhos_5b) >= 2 and SAM_RHO_50B in sam_rhos_5b:
        _cmap = plt.get_cmap(SAM_CMAP)
        _norm = mcolors.LogNorm(vmin=min(sam_rhos_5b), vmax=max(sam_rhos_5b))
        try:
            color_5e2 = _cmap(1.0 - _norm(SAM_RHO_50B))
        except Exception:
            pass
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for idx, midtrain_tokens in enumerate([5, 50]):
        ax = axes[idx]
        baseline_score = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, OLMES_EVAL_TASKS)
        sft_runs = filter_results(
            results,
            finetune_type="SFT",
            pretrain_token=pretrain_token,
            midtrain_token=midtrain_tokens,
        )
        sft_runs = [r for r in sft_runs if r.get("sft_lr") in SFT_LRS]
        adamw_runs = [r for r in sft_runs if r.get("optimizer") == "adamw"]
        sam_runs = [r for r in sft_runs if r.get("optimizer") == "sam"]
        sam_rhos = sorted(set(r.get("rho") for r in sam_runs if r.get("rho") is not None))

        adamw_scores = [
            _relative_forgetting_pct(baseline_score, _sft_score_at_lr(adamw_runs, lr, OLMES_EVAL_TASKS))
            for lr in lrs
        ]
        valid_a = [(x, y) for x, y in zip(lr_labels, adamw_scores) if y is not None]
        if valid_a:
            x_num = [lr for lr, y in zip(lrs, adamw_scores) if y is not None]
            y_num = [y for y in adamw_scores if y is not None]
            ax.plot(x_num, y_num, marker="o", color=ADAMW_COLOR, label="AdamW", linewidth=2, alpha=ALPHA)
            x_hull, y_hull = _convex_hull_boundary(x_num, y_num, lr_vals=x_num)
            if len(x_hull) >= 2:
                ax.plot(x_hull, y_hull, color=ADAMW_COLOR, alpha=0.8, linewidth=1.5, zorder=0)
        if sam_rhos:
            if len(sam_rhos) >= 2:
                cmap = plt.get_cmap(SAM_CMAP)
                norm = mcolors.LogNorm(vmin=min(sam_rhos), vmax=max(sam_rhos))
            for rho in sam_rhos:
                subset = [r for r in sam_runs if r.get("rho") == rho]
                scores = [
                    _relative_forgetting_pct(baseline_score, _sft_score_at_lr(subset, lr, OLMES_EVAL_TASKS))
                    for lr in lrs
                ]
                valid = [(x, y) for x, y in zip(lr_labels, scores) if y is not None]
                if valid:
                    x_num = [lr for lr, y in zip(lrs, scores) if y is not None]
                    y_num = [y for y in scores if y is not None]
                    if rho == SAM_RHO_50B:
                        color = color_5e2
                    else:
                        color = cmap(1.0 - norm(rho)) if len(sam_rhos) >= 2 else COLOR_MAP["sam"]
                    ax.plot(x_num, y_num, marker="s", color=color, label=f"SAM ρ={_rho_label(rho)}", linewidth=2, alpha=ALPHA)
                    x_hull, y_hull = _convex_hull_boundary(x_num, y_num, lr_vals=x_num)
                    if len(x_hull) >= 2:
                        ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5,  zorder=4)
        ax.set_xlabel("SFT learning rate", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
        ax.set_title(f"SFT eval (4T, {midtrain_tokens}B)", fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.set_xticks(lrs)
        ax.set_xticklabels(lr_labels, rotation=45)
        if idx == 0:
            ax.invert_xaxis()
        ax.grid(True, alpha=GRID_ALPHA)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    if handles:
        fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"] - 1, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("4T SFT eval (avg OLMES)", fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def _plot_sft_eval_by_group(results, out_path, pretrain_token=4, midtrain_tokens=5):
    """1x3 subplots (mcq, generative, heldout): avg score vs increasing LR for AdamW + SAM (all rhos)."""
    sft_runs = filter_results(
        results,
        finetune_type="SFT",
        pretrain_token=pretrain_token,
        midtrain_token=midtrain_tokens,
    )
    sft_runs = [r for r in sft_runs if r.get("sft_lr") in SFT_LRS]
    adamw_runs = [r for r in sft_runs if r.get("optimizer") == "adamw"]
    sam_runs = [r for r in sft_runs if r.get("optimizer") == "sam"]
    sam_rhos = sorted(set(r.get("rho") for r in sam_runs if r.get("rho") is not None))

    lrs = sorted(SFT_LRS)
    lr_labels = [_lr_label(lr) for lr in lrs]
    groups = list(OLMES_EVAL_GROUPS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    use_sam_cmap = len(sam_rhos) >= 2
    if use_sam_cmap:
        cmap = plt.get_cmap(SAM_CMAP)
        norm = mcolors.LogNorm(vmin=min(sam_rhos), vmax=max(sam_rhos))
    for col, group_name in enumerate(groups):
        ax = axes[col]
        task_list = OLMES_EVAL_GROUPS[group_name]
        baseline_score = _baseline_olmes_avg_score(results, pretrain_token, midtrain_tokens, task_list)
        adamw_scores = [
            _relative_forgetting_pct(baseline_score, _sft_score_at_lr(adamw_runs, lr, task_list))
            for lr in lrs
        ]
        valid_a = [(x, y) for x, y in zip(lr_labels, adamw_scores) if y is not None]
        if valid_a:
            x_num = [lr for lr, y in zip(lrs, adamw_scores) if y is not None]
            y_num = [y for y in adamw_scores if y is not None]
            ax.plot(x_num, y_num, marker="o", color=ADAMW_COLOR, label="AdamW", linewidth=2, alpha=ALPHA)
            x_hull, y_hull = _convex_hull_boundary(x_num, y_num, lr_vals=x_num)
            if len(x_hull) >= 2:
                ax.plot(x_hull, y_hull, color=ADAMW_COLOR, alpha=0.8, linewidth=1.5, zorder=0)
        for rho in sam_rhos:
            subset = [r for r in sam_runs if r.get("rho") == rho]
            scores = [
                _relative_forgetting_pct(baseline_score, _sft_score_at_lr(subset, lr, task_list))
                for lr in lrs
            ]
            valid = [(x, y) for x, y in zip(lr_labels, scores) if y is not None]
            if valid:
                x_num = [lr for lr, y in zip(lrs, scores) if y is not None]
                y_num = [y for y in scores if y is not None]
                color = cmap(1.0 - norm(rho)) if use_sam_cmap else COLOR_MAP["sam"]
                ax.plot(x_num, y_num, marker="s", color=color, label=f"SAM ρ={_rho_label(rho)}", linewidth=2, alpha=ALPHA)
                x_hull, y_hull = _convex_hull_boundary(x_num, y_num, lr_vals=x_num)
                if len(x_hull) >= 2:
                    ax.plot(x_hull, y_hull, color=color, alpha=1.0, linewidth=2.5,  zorder=4)
        ax.set_xlabel("SFT learning rate", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Relative forgetting (%)", fontsize=FONTSIZE["AXIS"])
        ax.set_title(group_name, fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.set_xticks(lrs)
        ax.set_xticklabels(lr_labels, rotation=45)
        ax.grid(True, alpha=GRID_ALPHA)
    handles, labels = axes[0].get_legend_handles_labels()
    ncol_leg = min(len(labels), 5) if labels else 4
    fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"] - 1, loc="upper center", ncol=ncol_leg, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"SFT eval (4T, {midtrain_tokens}B) by group", fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    plt.tight_layout()
    png_path = out_path[:-4] + ".png" if out_path.endswith(".pdf") else out_path
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    results = parse_results()
    base_out_dir = os.path.join(RESULTS_DIR, "plots", "cpt_pareto")
    out_effect_dir = os.path.join(base_out_dir, "effect")
    out_olmes_avg_dir = os.path.join(base_out_dir, "olmes_avg")
    out_olmes_group_dir = os.path.join(base_out_dir, "olmes_group")
    out_olmes_gsm8k_dir = os.path.join(base_out_dir, "olmes_gsm8k")
    out_task_metrics_dir = os.path.join(base_out_dir, "task_metrics")
    out_cpt_combined_dir = os.path.join(base_out_dir, "combined_cpt")
    out_sft_dir = os.path.join(base_out_dir, "sft_eval")
    out_sft_combined_dir = os.path.join(out_sft_dir, "combined")
    for d in [
        out_effect_dir,
        out_olmes_avg_dir,
        out_olmes_group_dir,
        out_olmes_gsm8k_dir,
        out_task_metrics_dir,
        out_cpt_combined_dir,
        out_sft_dir,
        out_sft_combined_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    # Plot 1: Effect of datasets — 4T, 5B; 6 datasets; SAM (all rho) + AdamW; reverse x; share x-axis across all
    _plot_pareto_2x3_avg(
        results,
        midtrain_tokens=5,
        out_path=os.path.join(out_effect_dir, "cpt_pareto_effect_of_datasets_4T_5B.pdf"),
        tasks=EFFECT_OF_DATASETS_TASKS,
        show_tokens_in_title=True,
        all_sam_rhos=True,
        pretrain_token=4,
        share_x_all=True,
    )

    # Effect of pretraining tokens (2T vs 4T): 1x2 avg (shared x,y) and 2x3 avg (shared x,y per column)
    _plot_effect_of_pretrain_1x2(
        results,
        out_path=os.path.join(out_effect_dir, "cpt_pareto_effect_of_pretrain_1x2_avg.pdf"),
        midtrain_tokens=5,
        all_sam_rhos=True,
    )
    _plot_effect_of_pretrain_2x3(
        results,
        out_path=os.path.join(out_effect_dir, "cpt_pareto_effect_of_pretrain_2x3_avg.pdf"),
        midtrain_tokens=5,
        all_sam_rhos=True,
    )

    # Plot 2: Effect of finetuning tokens — 4T, 5B; tulu/stackmathqa/musicpile at 20M and 50M; share axes
    _plot_pareto_2x3_avg(
        results,
        midtrain_tokens=5,
        out_path=os.path.join(out_effect_dir, "cpt_pareto_effect_of_tokens_4T_5B.pdf"),
        tasks=EFFECT_OF_TOKENS_TASKS,
        show_tokens_in_title=True,
        share_axes_col=True,
        all_sam_rhos=True,
        pretrain_token=4,
    )

    # Plot 3 (avg) & 4 (group): 4T, 5B, Tulu 50M — OLMES Pareto, AdamW vs SAM (multiple rho)
    _plot_pareto_olmes_eval_avg(
        results,
        midtrain_tokens=5,
        datasets=[("tulu", 50)],
        title_prefix="4T 5B",
        out_path=os.path.join(out_olmes_avg_dir, "cpt_pareto_olmes_avg_4T_5B_tulu50M.pdf"),
        pretrain_token=4,
        all_sam_rhos=True,
    )
    _plot_pareto_olmes_eval_single_task(
        results,
        midtrain_tokens=5,
        datasets=[("tulu", 50)],
        title_prefix="4T 5B",
        out_path=os.path.join(out_olmes_gsm8k_dir, "cpt_pareto_olmes_gsm8k_4T_5B_tulu50M.pdf"),
        pretrain_token=4,
        all_sam_rhos=True,
        task_name="gsm8k",
    )
    _plot_pareto_olmes_eval_by_group(
        results,
        midtrain_tokens=5,
        datasets=[("tulu", 50)],
        title_prefix="4T 5B",
        out_path=os.path.join(out_olmes_group_dir, "cpt_pareto_olmes_group_4T_5B_tulu50M.pdf"),
        pretrain_token=4,
        all_sam_rhos=True,
    )

    # Plot 5 (avg) & 6 (group): 4T, 50B, Tulu 50M — AdamW vs SAM (rho 5e-2)
    _plot_pareto_olmes_eval_avg(
        results,
        midtrain_tokens=50,
        datasets=[("tulu", 50)],
        title_prefix="4T 50B",
        out_path=os.path.join(out_olmes_avg_dir, "cpt_pareto_olmes_avg_4T_50B_tulu50M.pdf"),
        pretrain_token=4,
        sam_rhos_list=[5e-2],
    )
    _plot_pareto_olmes_eval_single_task(
        results,
        midtrain_tokens=50,
        datasets=[("tulu", 50)],
        title_prefix="4T 50B",
        out_path=os.path.join(out_olmes_gsm8k_dir, "cpt_pareto_olmes_gsm8k_4T_50B_tulu50M.pdf"),
        pretrain_token=4,
        sam_rhos_list=[5e-2],
        task_name="gsm8k",
    )
    _plot_pareto_olmes_eval_by_group(
        results,
        midtrain_tokens=50,
        datasets=[("tulu", 50)],
        title_prefix="4T 50B",
        out_path=os.path.join(out_olmes_group_dir, "cpt_pareto_olmes_group_4T_50B_tulu50M.pdf"),
        pretrain_token=4,
        sam_rhos_list=[5e-2],
    )

    # 4T 5B vs 50B combined 1x2 (shared x, y)
    _plot_pareto_olmes_4T_5B_50B_1x2(
        results,
        out_path=os.path.join(out_cpt_combined_dir, "cpt_pareto_olmes_4T_5B_50B_1x2.pdf"),
    )

    # Plot 7 (avg) & 8 (group): 2T, 5B, Tulu 50M — AdamW vs SAM (multiple rho)
    _plot_pareto_olmes_eval_avg(
        results,
        midtrain_tokens=5,
        datasets=[("tulu", 50)],
        title_prefix="2T 5B",
        out_path=os.path.join(out_olmes_avg_dir, "cpt_pareto_olmes_avg_2T_5B_tulu50M.pdf"),
        pretrain_token=2,
        all_sam_rhos=True,
    )
    _plot_pareto_olmes_eval_single_task(
        results,
        midtrain_tokens=5,
        datasets=[("tulu", 50)],
        title_prefix="2T 5B",
        out_path=os.path.join(out_olmes_gsm8k_dir, "cpt_pareto_olmes_gsm8k_2T_5B_tulu50M.pdf"),
        pretrain_token=2,
        all_sam_rhos=True,
        task_name="gsm8k",
    )
    _plot_pareto_olmes_eval_by_group(
        results,
        midtrain_tokens=5,
        datasets=[("tulu", 50)],
        title_prefix="2T 5B",
        out_path=os.path.join(out_olmes_group_dir, "cpt_pareto_olmes_group_2T_5B_tulu50M.pdf"),
        pretrain_token=2,
        all_sam_rhos=True,
    )

    # Plot 9 (avg) & 10 (group): SFT — 4T, 5B; AdamW + SAM (all rhos); LRs 1e-5, 3e-5, 6e-5
    _plot_sft_eval_avg(results, os.path.join(out_sft_dir, "sft_eval_avg_4T_5B.pdf"), midtrain_tokens=5)
    _plot_sft_eval_by_group(results, os.path.join(out_sft_dir, "sft_eval_group_4T_5B.pdf"), midtrain_tokens=5)

    # SFT — 4T, 50B; AdamW + SAM (all rhos present)
    _plot_sft_eval_avg(results, os.path.join(out_sft_dir, "sft_eval_avg_4T_50B.pdf"), midtrain_tokens=50)
    _plot_sft_eval_by_group(results, os.path.join(out_sft_dir, "sft_eval_group_4T_50B.pdf"), midtrain_tokens=50)

    # SFT 4T 5B vs 50B combined 1x2 (shared x, y)
    _plot_sft_eval_4T_5B_50B_1x2(results, os.path.join(out_sft_combined_dir, "sft_eval_4T_5B_50B_1x2.pdf"))


if __name__ == "__main__":
    main()
