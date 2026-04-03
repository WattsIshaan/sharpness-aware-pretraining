"""
Pareto plot for 60M model: CPT loss (starcoder_val) vs average downstream task metrics.
Config: 60m, 192B token, starcoder CPT 10M, cosine scheduler.
Uses final_results.json (with optional task metrics from ModelEvaluationDownstreamOLMo).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.config_globals import RESULTS_DIR, DOWNSTREAM_TASKS
from utils.plotting_globals import FONTSIZE, ALPHA, GRID_ALPHA

# Config filter
SIZE = 60
TOKEN = 192
CPT_DATASET = "starcoder"
CPT_TOKENS = 10
PRETRAIN_LRS = "cosine"

ADAMW_COLOR = "red"
SAM_COLOR = "C0"


def load_results():
    path = os.path.join(RESULTS_DIR, "final_results.json")
    with open(path, "r") as f:
        return json.load(f)


def filter_runs(results):
    out = []
    for r in results:
        if r.get("run_type") != "cpt":
            continue
        if r.get("size") != SIZE or r.get("token") != TOKEN:
            continue
        if r.get("cpt_dataset") != CPT_DATASET or r.get("cpt_tokens") != CPT_TOKENS:
            continue
        if r.get("pretrain_lrs") != PRETRAIN_LRS:
            continue
        if CPT_DATASET + "_val" not in r or r.get(CPT_DATASET + "_val") is None:
            continue
        vals = [r.get(t) for t in DOWNSTREAM_TASKS if t in r and r.get(t) is not None]
        if len(vals) != len(DOWNSTREAM_TASKS):
            continue
        out.append(r)
    return out


def convex_hull_boundary(x_vals, y_vals, lr_vals=None):
    """
    Return convex hull boundary as (x_list, y_list). Points in same order as input.
    If lr_vals is provided, exclude the edge between lowest-lr and highest-lr vertices
    (Pareto frontier / open boundary). Otherwise return closed hull.
    """
    x_vals = list(x_vals)
    y_vals = list(y_vals)
    n = len(x_vals)
    if n == 0:
        return [], []
    if n == 1:
        return x_vals, y_vals
    if n == 2:
        return x_vals, y_vals
    try:
        from scipy.spatial import ConvexHull
        points = np.column_stack([np.asarray(x_vals), np.asarray(y_vals)])
        hull = ConvexHull(points)
        vert_idx = hull.vertices  # indices into points, ccw order
        nv = len(vert_idx)
        if lr_vals is None or len(lr_vals) != n:
            hull_pts = points[vert_idx]
            x_hull = hull_pts[:, 0].tolist()
            y_hull = hull_pts[:, 1].tolist()
            return x_hull, y_hull
        lr_at_vert = np.array([float(lr_vals[i]) for i in vert_idx])
        pos_min = int(np.argmin(lr_at_vert))
        pos_max = int(np.argmax(lr_at_vert))
        # Exclude edge between min-lr and max-lr: traverse the long way
        step = 1 if (pos_min + 1) % nv != pos_max else -1
        order = []
        k = pos_min
        for _ in range(nv):
            order.append(k)
            k = (k + step) % nv
        hull_pts = points[vert_idx[np.array(order)]]
        x_hull = hull_pts[:, 0].tolist()
        y_hull = hull_pts[:, 1].tolist()

        # Truncate polyline once x changes direction (removes future edges).
        if len(x_hull) >= 3:
            eps = 1e-12
            direction = 0  # -1 decreasing, +1 increasing
            for i in range(len(x_hull) - 1):
                dx = x_hull[i + 1] - x_hull[i]
                if abs(dx) > eps:
                    direction = -1 if dx < 0 else 1
                    break
            if direction != 0:
                keep = [0]
                for i in range(len(x_hull) - 1):
                    dx = x_hull[i + 1] - x_hull[i]
                    if direction < 0:
                        if dx <= eps:
                            keep.append(i + 1)
                        else:
                            break
                    else:
                        if dx >= -eps:
                            keep.append(i + 1)
                        else:
                            break
                x_hull = [x_hull[i] for i in keep]
                y_hull = [y_hull[i] for i in keep]

        return x_hull, y_hull
    except Exception:
        if n >= 2:
            order = np.argsort(np.array(x_vals))
            return [x_vals[i] for i in order], [y_vals[i] for i in order]
        return [], []


def main():
    results = load_results()
    runs = filter_runs(results)
    if not runs:
        print("No runs found for 60m 192B starcoder 10M cosine. Run preprocess_results.py and ensure ModelEvaluationDownstreamOLMo has matching downstream evals.")
        return

    # Average task metrics: values from eval are typically 0-1, scale to 0-100
    def avg_task(r):
        v = [r[t] for t in DOWNSTREAM_TASKS]
        m = np.mean(v)
        return m * 100 if m <= 1.5 else m

    by_optim = {}
    for r in runs:
        opt = r.get("optimizer", "adamw")
        if opt not in by_optim:
            by_optim[opt] = []
        by_optim[opt].append(r)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for optim, group in by_optim.items():
        group = sorted(group, key=lambda r: r.get("cpt_lr") or 0)
        x_vals = [avg_task(r) for r in group]
        y_vals = [r[CPT_DATASET + "_val"] for r in group]
        lr_vals = [r.get("cpt_lr") for r in group]
        color = ADAMW_COLOR if optim == "adamw" else SAM_COLOR
        label = "AdamW" if optim == "adamw" else "SAM"
        ax.scatter(x_vals, y_vals, color=color, label=label, alpha=ALPHA, s=60)
        if len(x_vals) >= 2:
            x_h, y_h = convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
            if len(x_h) >= 2:
                ax.plot(x_h, y_h, color=color, alpha=0.8, linewidth=1.5, zorder=0)

    ax.set_xlabel("Average Accuracy Score", fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(f"CPT loss ({CPT_DATASET})", fontsize=FONTSIZE["AXIS"])
    ax.set_title(f"SFT on Starcoder vs Accuracy Eval", fontsize=FONTSIZE["TITLE"])
    ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
    ax.invert_xaxis()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=FONTSIZE["LEGEND"] - 1)
    plt.tight_layout()
    out_dir = os.path.join(RESULTS_DIR, "plots", "60m_pareto")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "60m_pareto_ds.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # 2x3 per-task Pareto: one subplot per downstream task (x = task score, y = CPT loss)
    def task_score(r, task):
        v = r.get(task)
        if v is None:
            return None
        return v * 100 if v <= 1.5 else v

    nrows, ncols = 2, 3
    fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey="col")
    for idx, task in enumerate(DOWNSTREAM_TASKS):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        for optim, group in by_optim.items():
            group = sorted(group, key=lambda r: r.get("cpt_lr") or 0)
            x_vals = [task_score(r, task) for r in group]
            if any(x is None for x in x_vals):
                continue
            y_vals = [r[CPT_DATASET + "_val"] for r in group]
            lr_vals = [r.get("cpt_lr") for r in group]
            color = ADAMW_COLOR if optim == "adamw" else SAM_COLOR
            label = "AdamW" if optim == "adamw" else "SAM"
            ax.scatter(x_vals, y_vals, color=color, label=label, alpha=ALPHA, s=60)
            if len(x_vals) >= 2:
                x_h, y_h = convex_hull_boundary(x_vals, y_vals, lr_vals=lr_vals)
                if len(x_h) >= 2:
                    ax.plot(x_h, y_h, color=color, alpha=0.8, linewidth=1.5, zorder=0)
        ax.set_xlabel("Score", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel(f"CPT loss ({CPT_DATASET})", fontsize=FONTSIZE["AXIS"])
        title = task.replace("_", " ").title()
        ax.set_title(title, fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.invert_xaxis()
        ax.grid(True, alpha=GRID_ALPHA)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, fontsize=FONTSIZE["LEGEND"] - 1, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.02))
    fig2.suptitle(f"60M 192B CPT {CPT_DATASET} 10M cosine — per task", fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    plt.tight_layout()
    out_path_2x3 = os.path.join(out_dir, "60m_pareto_ds_per_task.png")
    plt.savefig(out_path_2x3, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path_2x3}")


if __name__ == "__main__":
    main()
