"""
Export 1B midtrain OLMES scores to Excel (4T pretrain, 50B midtrain, SAM ρ=5e-2).

Per CPT dataset (same list as ``pareto_1b_thresholded``), one sheet with six rows:

1. AdamW (Base)
2. SAM (Base)
3. AdamW (4-bit)
4. SAM (4-bit)
5. AdamW (SFT)
6. SAM (SFT)

Columns: ``Setting``, ``Average`` (mean over all OLMES tasks when complete), then
per-task scores with headers from ``BASE_EVAL_TASK_MAP`` in ``utils/plotting_globals``.

Run from ``plotting/``::

    PYTHONPATH=. python table_1b.py

Requires: scipy, pandas, openpyxl.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from utils.config_globals import RESULTS_DIR
from utils.plotting_globals import BASE_EVAL_TASK_MAP
from preprocess_midtrain_olmo import (
    OLMES_EVAL_TASKS,
    get_cpt_run_info,
    get_olmo_run_info,
    parse_results,
)
from pareto_1b import (
    _convex_hull_boundary,
    _cpt_lr_window,
    _mean_olmes_tasks_if_complete,
    _pick_downstream_baseline_run,
    _series_for_technique_olmes_avg,
)
from pareto_1b_thresholded import (
    LOSS_MARGIN_4T_50B_BY_DATASET,
    THRESHOLDED_OLMES_LOSS_4T_50B_50M,
    _pick_sam_downstream_baseline_run,
)

PRETRAIN_TOKEN = 4
MIDTRAIN_TOKENS = 50
SAM_RHO = 5e-2
BATCH_SIZE = 1024
DEFAULT_LOSS_MARGIN = 0.005


def _open_hull_index_path(x_vals, y_vals, lr_vals) -> list[int]:
    """Indices into the original series along the same open hull path as ``_convex_hull_boundary``."""
    points = np.column_stack([np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)])
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    hull = ConvexHull(points)
    vert_idx = hull.vertices
    nv = len(vert_idx)
    lr_at_vert = np.array([float(lr_vals[i]) for i in vert_idx])
    pos_min = int(np.argmin(lr_at_vert))
    pos_max = int(np.argmax(lr_at_vert))
    step = 1 if (pos_min + 1) % nv != pos_max else -1
    order: list[int] = []
    k = pos_min
    for _ in range(nv):
        order.append(k)
        k = (k + step) % nv
    return [int(vert_idx[j]) for j in order]


def _series_with_oe(runs, optim, rho, cpt_dataset, task_list, lr_range=None):
    """Like ``_series_for_technique_olmes_avg`` but also return ``olmes_eval`` dict per point."""
    lo, hi = _cpt_lr_window(cpt_dataset, lr_range)
    subset = [r for r in runs if r.get("optimizer") == optim and r.get("rho") == rho]
    subset = sorted(subset, key=lambda r: float(r.get("cpt_lr") or 0))
    x, y, lrs, oes = [], [], [], []
    for r in subset:
        oe = r.get("olmes_eval") or {}
        x_mean = _mean_olmes_tasks_if_complete(oe, task_list)
        if x_mean is None or r.get("finetuning_val_loss") is None:
            continue
        if not (lo <= r.get("cpt_lr") <= hi):
            continue
        x.append(x_mean)
        y.append(float(r["finetuning_val_loss"]))
        lrs.append(r.get("cpt_lr"))
        oes.append(oe)
    return x, y, lrs, oes


def _interpolate_olmes_at_y(
    idx_path: list[int],
    y_line: float,
    y_series: list[float],
    oes: list[dict],
    task_list: list[str],
    eps: float = 1e-12,
) -> dict | None:
    """Linearly interpolate each task along a hull edge where ``y_line`` crosses."""
    n = len(idx_path)
    if n < 2:
        return None

    def between(i0: int, i1: int) -> dict | None:
        y0, y1 = y_series[i0], y_series[i1]
        oe0, oe1 = oes[i0], oes[i1]
        lo, hi = min(y0, y1), max(y0, y1)
        if y_line < lo - eps or y_line > hi + eps:
            return None
        if abs(y1 - y0) < 1e-15:
            if abs(y0 - y_line) < 1e-9:
                m0 = _mean_olmes_tasks_if_complete(oe0, task_list)
                m1 = _mean_olmes_tasks_if_complete(oe1, task_list)
                if m0 is None:
                    return oe1
                if m1 is None:
                    return oe0
                return oe0 if m0 >= m1 else oe1
            return None
        u = (y_line - y0) / (y1 - y0)
        if not (-eps <= u <= 1.0 + eps):
            return None
        out: dict = {}
        for t in task_list:
            v0 = oe0.get(t)
            v1 = oe1.get(t)
            if v0 is None or v1 is None:
                return None
            out[t] = float(v0) + u * (float(v1) - float(v0))
        return out

    for k in range(n - 1):
        r = between(idx_path[k], idx_path[k + 1])
        if r is not None:
            return r
    r = between(idx_path[-1], idx_path[0])
    if r is not None:
        return r
    return None


def _nearest_run_oe_by_loss(
    y_series: list[float], oes: list[dict], y_target: float, task_list: list[str]
) -> dict | None:
    if not y_series:
        return None
    best_i = min(range(len(y_series)), key=lambda i: abs(y_series[i] - y_target))
    oe = oes[best_i]
    if _mean_olmes_tasks_if_complete(oe, task_list) is None:
        return None
    return oe


def _oe_downstream_baseline_adamw(results, task_list) -> dict | None:
    base = _pick_downstream_baseline_run(
        results, PRETRAIN_TOKEN, MIDTRAIN_TOKENS, prefer_eval_types=("hf", "olmo")
    )
    if base is None:
        return None
    oe = base.get("olmes_eval") or {}
    return oe if _mean_olmes_tasks_if_complete(oe, task_list) is not None else None


def _oe_downstream_baseline_sam(results, task_list) -> dict | None:
    base = _pick_sam_downstream_baseline_run(
        results, PRETRAIN_TOKEN, MIDTRAIN_TOKENS, SAM_RHO, prefer_eval_types=("hf", "olmo")
    )
    if base is None:
        return None
    oe = base.get("olmes_eval") or {}
    return oe if _mean_olmes_tasks_if_complete(oe, task_list) is not None else None


def _oe_hf_4bit(results, optim, rho, task_list) -> dict | None:
    kw = dict(
        pretrain_token=PRETRAIN_TOKEN,
        midtrain_tokens=MIDTRAIN_TOKENS,
        batch_size=BATCH_SIZE,
    )
    matched = get_olmo_run_info(
        results,
        optim=optim,
        rho=rho if optim == "sam" else None,
        eval_type="hf-4bit",
        **kw,
    )
    if not matched:
        return None
    for r in matched:
        oe = r.get("olmes_eval") or {}
        if _mean_olmes_tasks_if_complete(oe, task_list) is not None:
            return oe
    return None


def _thresholded_olmes_row(
    runs,
    optim,
    rho,
    cpt_dataset: str,
    task_list: list[str],
    loss_margin: float,
) -> dict | None:
    x_a, y_a, lr_a = _series_for_technique_olmes_avg(runs, "adamw", None, cpt_dataset, task_list)
    if not y_a:
        return None
    y_line = float(min(y_a)) * (1.0 + loss_margin)

    x_s, y_s, lrs_s, oes = _series_with_oe(runs, optim, rho, cpt_dataset, task_list)
    if len(x_s) < 2 or len(lrs_s) != len(oes):
        return None

    xh, yh = _convex_hull_boundary(x_s, y_s, lrs_s)
    if len(xh) < 2:
        return None

    idx_path = _open_hull_index_path(x_s, y_s, lrs_s)
    interp = _interpolate_olmes_at_y(idx_path, y_line, y_s, oes, task_list)
    if interp is not None:
        return interp
    return _nearest_run_oe_by_loss(y_s, oes, y_line, task_list)


def _task_column_headers() -> list[str]:
    """Display names for OLMES task columns (``BASE_EVAL_TASK_MAP``), ``OLMES_EVAL_TASKS`` order."""
    return [BASE_EVAL_TASK_MAP.get(t, t) for t in OLMES_EVAL_TASKS]


def _detail_column_names() -> list[str]:
    return ["Setting", "Average"] + _task_column_headers()


def _row_to_record_flat(setting: str, oe: dict | None, task_list: list[str]) -> dict:
    rec: dict = {"Setting": setting, "Average": np.nan}
    for t in OLMES_EVAL_TASKS:
        col = BASE_EVAL_TASK_MAP.get(t, t)
        if oe is None:
            rec[col] = np.nan
        else:
            v = oe.get(t)
            rec[col] = float(v) if v is not None else np.nan
    if oe is not None:
        m = _mean_olmes_tasks_if_complete(oe, task_list)
        rec["Average"] = float(m) if m is not None else np.nan
    return rec


def build_sheet_records(results, cpt_dataset: str, cpt_tokens: int, loss_margin: float) -> list[dict]:
    task_list = list(OLMES_EVAL_TASKS)
    runs = get_cpt_run_info(
        results,
        midtrain_tokens=MIDTRAIN_TOKENS,
        cpt_dataset=cpt_dataset,
        cpt_tokens=cpt_tokens,
        pretrain_token=PRETRAIN_TOKEN,
        step=-1,
    )
    if not runs:
        runs = []

    rows: list[dict] = []
    rows.append(
        _row_to_record_flat(
            "AdamW (Base)",
            _oe_downstream_baseline_adamw(results, task_list),
            task_list,
        )
    )
    rows.append(
        _row_to_record_flat(
            "SAM (Base)",
            _oe_downstream_baseline_sam(results, task_list),
            task_list,
        )
    )
    rows.append(
        _row_to_record_flat(
            "AdamW (4-bit)",
            _oe_hf_4bit(results, "adamw", None, task_list),
            task_list,
        )
    )
    rows.append(
        _row_to_record_flat(
            "SAM (4-bit)",
            _oe_hf_4bit(results, "sam", SAM_RHO, task_list),
            task_list,
        )
    )
    rows.append(
        _row_to_record_flat(
            "AdamW (SFT)",
            _thresholded_olmes_row(runs, "adamw", None, cpt_dataset, task_list, loss_margin)
            if runs
            else None,
            task_list,
        )
    )
    rows.append(
        _row_to_record_flat(
            "SAM (SFT)",
            _thresholded_olmes_row(runs, "sam", SAM_RHO, cpt_dataset, task_list, loss_margin)
            if runs
            else None,
            task_list,
        )
    )
    return rows


def write_excel(out_path: str, loss_margin: float | None = None) -> None:
    results = parse_results()
    colnames = _detail_column_names()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for cpt_dataset, cpt_tokens in THRESHOLDED_OLMES_LOSS_4T_50B_50M:
            m = (
                loss_margin
                if loss_margin is not None
                else LOSS_MARGIN_4T_50B_BY_DATASET.get(cpt_dataset, DEFAULT_LOSS_MARGIN)
            )
            recs = build_sheet_records(results, cpt_dataset, cpt_tokens, m)
            df = pd.DataFrame(recs, columns=colnames)
            sheet = cpt_dataset[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Wrote {out_path}")


def main():
    out_dir = os.path.join(RESULTS_DIR, "plots", "cpt_pareto", "thresholded_1b")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "olmes_1b_table.xlsx")
    write_excel(out_path)


if __name__ == "__main__":
    main()
