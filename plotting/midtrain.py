import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config_globals import RESULTS_DIR
from utils.plotting_globals import (
    BASE_EVAL_TASK_MAP, SFT_EVAL_TASK_MAP,
    COLOR_MAP, OPTIM_MAP, FONTSIZE, MARKERS, ALPHA, GRID_ALPHA,
)
from preprocess_midtrain import (
    parse_results,
    get_midtrain_run_info,
    _clean_task_name,
    BASE_MCQ_TASKS,
    BASE_GENERATIVE_TASKS,
    BASE_HELDOUT_TASKS,
    SFT_KNOWLEDGE_TASKS,
    SFT_REASONING_TASKS,
    SFT_CODE_TASKS,
    SFT_IF_TASKS,
)


def _flatten_task_metrics(run):
    """Return a flat dict of clean_task_name → score from a single run_info."""
    flat = {}
    for group in run["task_metrics"].values():
        if isinstance(group, dict):
            flat.update(group)
    return flat


def _rho_label(rho):
    """Format a rho float as a compact scientific-notation string."""
    coeff, exp = f"{rho:.2e}".replace("+", "").split("e")
    coeff = coeff.rstrip("0").rstrip(".")
    return f"{coeff}e{exp}"


# ---------------------------------------------------------------------------
# Base Model table
# ---------------------------------------------------------------------------

def build_base_excel(results):
    """Build a DataFrame for Base Model evals.

    Rows   : OLMo Base, AdamW, SAM, AdamW 4-bit, SAM 4-bit
    Columns: MCQ | Generative | Heldout tasks
    """

    row_configs = [
        ("OLMo",   dict(optim="olmo",  run_type="midtrain")),
        ("AdamW",       dict(optim="adamw", run_type="midtrain")),
        ("SAM",         dict(optim="sam",   run_type="midtrain")),
        ("AdamW 4-bit", dict(optim="adamw", run_type="quant", quant_bit=4)),
        ("SAM 4-bit",   dict(optim="sam",   run_type="quant", quant_bit=4)),
    ]

    # Column order mirrors the grouping in the original build_quant_excel
    col_order = (
        [_clean_task_name(t) for t in BASE_MCQ_TASKS]
        + [_clean_task_name(t) for t in BASE_GENERATIVE_TASKS]
        + [_clean_task_name(t) for t in BASE_HELDOUT_TASKS]
    )

    display_cols = [BASE_EVAL_TASK_MAP.get(c, c) for c in col_order]

    group_labels = (
        ["MCQ"] * len(BASE_MCQ_TASKS)
        + ["Generative"] * len(BASE_GENERATIVE_TASKS)
        + ["Heldout"] * len(BASE_HELDOUT_TASKS)
    )
    columns = pd.MultiIndex.from_arrays(
        [group_labels, display_cols], names=["Group", "Task"]
    )

    rows = []
    index = []
    for label, kwargs in row_configs:
        matched = get_midtrain_run_info(results, **kwargs)
        if matched is None or len(matched) == 0:
            print(f"Warning: no results for {label}")
            continue
        run = matched[0]
        flat = _flatten_task_metrics(run)
        rows.append([flat.get(col, None) for col in col_order])
        index.append(label)

    df = pd.DataFrame(rows, index=index, columns=columns)
    df[("Average", "Avg")] = df.mean(axis=1).round(1)

    out_path = os.path.join(RESULTS_DIR, "midtrain_base_eval_results.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Base Model Evals")
    print(f"Saved Base Model Excel to {out_path}")
    print(df.to_string())


# ---------------------------------------------------------------------------
# 5B Base Model table
# ---------------------------------------------------------------------------

SAM_5B_RHOS = [2e-2, 5e-2, 1e-1, 1.25e-1, 1.50e-1, 2e-1]
SAM_5B_ANNEAL_RHOS = [5e-2, 1e-1, 2e-1]
BATCH_SIZES_5B = [1024, 512]


def build_base_excel_5B(results):
    """Build a DataFrame for 5B midtrain Base Model evals.

    For 5B, SAM has multiple rho values and potentially different batch sizes.
    Rows include AdamW, SAM, SAM Anneal variants, plus 4-bit quantized counterparts.
    """

    row_configs = []

    for bs in BATCH_SIZES_5B:
        bs_tag = f" bs{bs}" if len(BATCH_SIZES_5B) > 1 else ""

        row_configs.append(
            (f"AdamW{bs_tag}",
             dict(optim="adamw", run_type="midtrain", midtrain_tokens=5, batch_size=bs)),
        )
        for rho in SAM_5B_RHOS:
            row_configs.append(
                (f"SAM rho={_rho_label(rho)}{bs_tag}",
                 dict(optim="sam", rho=rho, run_type="midtrain", midtrain_tokens=5, batch_size=bs)),
            )
        for rho in SAM_5B_ANNEAL_RHOS:
            row_configs.append(
                (f"SAM Anneal rho={_rho_label(rho)}{bs_tag}",
                 dict(optim="sam", rho=rho, run_type="midtrain", midtrain_tokens=5, batch_size=bs, anneal_sam=True)),
            )

        row_configs.append(
            (f"AdamW 4-bit{bs_tag}",
             dict(optim="adamw", run_type="quant", quant_bit=4, midtrain_tokens=5, batch_size=bs)),
        )
        for rho in SAM_5B_RHOS:
            row_configs.append(
                (f"SAM rho={_rho_label(rho)} 4-bit{bs_tag}",
                 dict(optim="sam", rho=rho, run_type="quant", quant_bit=4, midtrain_tokens=5, batch_size=bs)),
            )
        for rho in SAM_5B_ANNEAL_RHOS:
            row_configs.append(
                (f"SAM Anneal rho={_rho_label(rho)} 4-bit{bs_tag}",
                 dict(optim="sam", rho=rho, run_type="quant", quant_bit=4, midtrain_tokens=5, batch_size=bs, anneal_sam=True)),
            )

    col_order = (
        [_clean_task_name(t) for t in BASE_MCQ_TASKS]
        + [_clean_task_name(t) for t in BASE_GENERATIVE_TASKS]
        + [_clean_task_name(t) for t in BASE_HELDOUT_TASKS]
    )

    display_cols = [BASE_EVAL_TASK_MAP.get(c, c) for c in col_order]

    group_labels = (
        ["MCQ"] * len(BASE_MCQ_TASKS)
        + ["Generative"] * len(BASE_GENERATIVE_TASKS)
        + ["Heldout"] * len(BASE_HELDOUT_TASKS)
    )
    columns = pd.MultiIndex.from_arrays(
        [group_labels, display_cols], names=["Group", "Task"]
    )

    rows = []
    index = []
    for label, kwargs in row_configs:
        matched = get_midtrain_run_info(results, **kwargs)
        if matched is None or len(matched) == 0:
            continue
        run = matched[0]
        flat = _flatten_task_metrics(run)
        rows.append([flat.get(col, None) for col in col_order])
        index.append(label)

    if not rows:
        print("Warning: no 5B base eval results found")
        return

    df = pd.DataFrame(rows, index=index, columns=columns)
    df[("Average", "Avg")] = df.mean(axis=1).round(1)

    out_path = os.path.join(RESULTS_DIR, "midtrain_base_eval_5B_results.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="5B Base Model Evals")
    print(f"Saved 5B Base Model Excel to {out_path}")
    print(df.to_string())


# ---------------------------------------------------------------------------
# SFT Model table
# ---------------------------------------------------------------------------

LR_LABELS = {
    1e-5: "1e-5", 3e-5: "3e-5", 4e-5: "4e-5",
    6e-5: "6e-5", 8e-5: "8e-5", 1e-4: "1e-4",
}
SFT_LRS = [1e-5, 3e-5, 4e-5, 6e-5, 8e-5, 1e-4]


def build_sft_excel(results):
    """Build a DataFrame for SFT Model evals.

    Rows   : OLMo-SFT (lr=3e-5),
             AdamW  (lr = 1e-5, 3e-5, 6e-5, 1e-4),
             SAM    (lr = 1e-5, 3e-5, 6e-5, 1e-4)
    Columns: Knowledge | Reasoning | Code | Instruction Following
    """

    row_configs = [
        ("OLMo (lr=3e-5)", dict(optim="olmo", run_type="sft", sft_lr=3e-5)),
    ]
    for lr in SFT_LRS:
        row_configs.append(
            (f"AdamW (lr={LR_LABELS[lr]})",
             dict(optim="adamw", run_type="sft", sft_lr=lr)),
        )
    for lr in SFT_LRS:
        row_configs.append(
            (f"SAM (lr={LR_LABELS[lr]})",
             dict(optim="sam", run_type="sft", sft_lr=lr)),
        )

    col_order = (
        [_clean_task_name(t) for t in SFT_KNOWLEDGE_TASKS]
        + [_clean_task_name(t) for t in SFT_REASONING_TASKS]
        + [_clean_task_name(t) for t in SFT_CODE_TASKS]
        + [_clean_task_name(t) for t in SFT_IF_TASKS]
    )

    display_cols = [SFT_EVAL_TASK_MAP.get(c, c) for c in col_order]

    group_labels = (
        ["Knowledge"] * len(SFT_KNOWLEDGE_TASKS)
        + ["Reasoning"] * len(SFT_REASONING_TASKS)
        + ["Code"] * len(SFT_CODE_TASKS)
        + ["Instruction Following"] * len(SFT_IF_TASKS)
    )
    columns = pd.MultiIndex.from_arrays(
        [group_labels, display_cols], names=["Group", "Task"]
    )

    rows = []
    index = []
    for label, kwargs in row_configs:
        matched = get_midtrain_run_info(results, **kwargs)
        if matched is None or len(matched) == 0:
            # Missing results – fill with dashes
            rows.append(["—"] * len(col_order))
            index.append(label)
            continue
        run = matched[0]
        flat = _flatten_task_metrics(run)
        row = []
        for col in col_order:
            val = flat.get(col, None)
            row.append(val if val is not None else "—")
        rows.append(row)
        index.append(label)

    df = pd.DataFrame(rows, index=index, columns=columns)

    # Compute average only over numeric cells
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    df[("Average", "Avg")] = numeric_df.mean(axis=1).round(1)

    out_path = os.path.join(RESULTS_DIR, "midtrain_sft_eval_results.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="SFT Model Evals")
    print(f"Saved SFT Model Excel to {out_path}")
    print(df.to_string())


# ---------------------------------------------------------------------------
# 50B SFT Pareto: task1 vs task2 across LR for AdamW & SAM
# ---------------------------------------------------------------------------

ALL_SFT_TASKS = (
    [_clean_task_name(t) for t in SFT_KNOWLEDGE_TASKS]
    + [_clean_task_name(t) for t in SFT_REASONING_TASKS]
    + [_clean_task_name(t) for t in SFT_CODE_TASKS]
    + [_clean_task_name(t) for t in SFT_IF_TASKS]
)


def _collect_sft_optim_data(results):
    """Gather per-optimizer, per-LR flat task scores for 50B SFT runs."""
    optim_configs = {
        "adamw": dict(optim="adamw", run_type="sft", midtrain_tokens=50),
        "sam": dict(optim="sam", rho=5e-2, run_type="sft", midtrain_tokens=50),
    }
    optim_data = {}
    for optim_key, base_kwargs in optim_configs.items():
        runs = get_midtrain_run_info(results, **base_kwargs)
        if runs is None:
            continue
        task_by_lr = {}
        for run in runs:
            lr = run.get("sft_lr")
            flat = _flatten_task_metrics(run)
            task_by_lr[lr] = flat
        optim_data[optim_key] = task_by_lr
    return optim_data


def _plot_pareto_on_ax(ax, optim_data, task_x, task_y):
    """Draw AdamW & SAM pareto curves on a single Axes."""
    for idx, (optim_key, task_by_lr) in enumerate(optim_data.items()):
        sorted_lrs = sorted(task_by_lr.keys())
        xs, ys, lr_labels = [], [], []
        for lr in sorted_lrs:
            x_val = task_by_lr[lr].get(task_x)
            y_val = task_by_lr[lr].get(task_y)
            if x_val is not None and y_val is not None:
                xs.append(x_val)
                ys.append(y_val)
                lr_labels.append(lr)
        if not xs:
            continue
        ax.plot(
            xs, ys,
            marker=MARKERS[idx],
            color=COLOR_MAP[optim_key],
            label=OPTIM_MAP[optim_key],
            alpha=ALPHA,
        )
        for x, y, lr in zip(xs, ys, lr_labels):
            ax.annotate(
                f"{lr:.0e}", (x, y),
                textcoords="offset points", xytext=(4, 4),
                fontsize=6, color=COLOR_MAP[optim_key],
            )
    ax.invert_xaxis()
    ax.invert_yaxis()


KNOWLEDGE_TASKS = [_clean_task_name(t) for t in SFT_KNOWLEDGE_TASKS
                   if _clean_task_name(t) != "truthfulqa"]
REASONING_AND_CODE_TASKS = (
    [_clean_task_name(t) for t in SFT_REASONING_TASKS]
    + [_clean_task_name(t) for t in SFT_CODE_TASKS]
)


def build_sft_pareto_50B(results):
    """For each reasoning/code task (y), 1×3 subplots vs knowledge tasks (x)."""
    optim_data = _collect_sft_optim_data(results)
    if not optim_data:
        print("Warning: no 50B SFT results found for pareto")
        return

    out_dir = os.path.join(RESULTS_DIR, "plots/midtrain_pareto_sft_50B")
    os.makedirs(out_dir, exist_ok=True)

    ncols = len(KNOWLEDGE_TASKS)

    for task_y in REASONING_AND_CODE_TASKS:
        fig, axs = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4))
        if ncols == 1:
            axs = [axs]

        for col_idx, task_x in enumerate(KNOWLEDGE_TASKS):
            ax = axs[col_idx]
            _plot_pareto_on_ax(ax, optim_data, task_x, task_y)
            disp_x = SFT_EVAL_TASK_MAP.get(task_x, task_x)
            ax.set_xlabel(disp_x, fontsize=FONTSIZE["AXIS"])
            ax.set_title(disp_x, fontsize=FONTSIZE["TITLE"])
            ax.tick_params(axis="both", which="major",
                           labelsize=FONTSIZE["TICKS"])
            ax.grid(True, alpha=GRID_ALPHA)
            if col_idx == 0:
                disp_y = SFT_EVAL_TASK_MAP.get(task_y, task_y)
                ax.set_ylabel(disp_y, fontsize=FONTSIZE["AXIS"])

        disp_y = SFT_EVAL_TASK_MAP.get(task_y, task_y)
        fig.suptitle(f"{disp_y} vs Knowledge Tasks",
                     fontsize=FONTSIZE["TITLE"] + 2, y=1.02)

        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 0.99), ncol=len(handles),
                       fontsize=FONTSIZE["LEGEND"])

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{task_y}_vs_knowledge.pdf"),
                    bbox_inches="tight")
        plt.close()

    print(f"Saved {len(REASONING_AND_CODE_TASKS)} SFT pareto subplot "
          f"figures to {out_dir}")


# ---------------------------------------------------------------------------
# 5B Ablation Summary Plots
# ---------------------------------------------------------------------------

BASE_COL_ORDER = (
    [_clean_task_name(t) for t in BASE_MCQ_TASKS]
    + [_clean_task_name(t) for t in BASE_GENERATIVE_TASKS]
    + [_clean_task_name(t) for t in BASE_HELDOUT_TASKS]
)

BASE_GROUP_TASKS = {
    "MCQ": [_clean_task_name(t) for t in BASE_MCQ_TASKS],
    "Generative": [_clean_task_name(t) for t in BASE_GENERATIVE_TASKS],
    "Heldout": [_clean_task_name(t) for t in BASE_HELDOUT_TASKS],
}


def _avg_score(results, **kwargs):
    """Return overall average across all base eval tasks for a single config."""
    matched = get_midtrain_run_info(results, **kwargs)
    if not matched:
        return None
    flat = _flatten_task_metrics(matched[0])
    vals = [flat[t] for t in BASE_COL_ORDER if t in flat]
    return np.mean(vals) if vals else None


def _group_avgs(results, **kwargs):
    """Return {group_name: avg} for a single config."""
    matched = get_midtrain_run_info(results, **kwargs)
    if not matched:
        return {}
    flat = _flatten_task_metrics(matched[0])
    out = {}
    for gname, tasks in BASE_GROUP_TASKS.items():
        vals = [flat[t] for t in tasks if t in flat]
        out[gname] = np.mean(vals) if vals else None
    return out


def build_5B_ablation_summary(results):
    """Multi-panel summary of 5B midtrain ablation results."""

    out_dir = os.path.join(RESULTS_DIR, "plots/midtrain_5B_ablation")
    os.makedirs(out_dir, exist_ok=True)

    rho_values = [2e-2, 5e-2, 1e-1, 1.25e-1, 1.5e-1, 2e-1]
    anneal_rhos = [5e-2, 1e-1, 2e-1]
    bs = 1024

    # ── Panel 1: 4-bit Quantized Average vs rho ──
    adamw_q_avg = _avg_score(results, optim="adamw", run_type="quant",
                             quant_bit=4, midtrain_tokens=5, batch_size=bs)

    sam_q_avgs = [_avg_score(results, optim="sam", rho=r, run_type="quant",
                             quant_bit=4, midtrain_tokens=5, batch_size=bs)
                  for r in rho_values]
    anneal_q_avgs = [_avg_score(results, optim="sam", rho=r, run_type="quant",
                                quant_bit=4, midtrain_tokens=5, batch_size=bs,
                                anneal_sam=True)
                     for r in anneal_rhos]
    sam_bs512_q_avgs = [_avg_score(results, optim="sam", rho=r, run_type="quant",
                                   quant_bit=4, midtrain_tokens=5, batch_size=512)
                        for r in rho_values]

    rho_labels = [_rho_label(r) for r in rho_values]
    anneal_labels = [_rho_label(r) for r in anneal_rhos]

    def _group_score(group_name, **kwargs):
        """Return average score for a single task group given a config."""
        gavg = _group_avgs(results, **kwargs)
        return gavg.get(group_name)

    def _plot_rho_lines(ax, sam_vals, anneal_vals, bs512_vals, adamw_val,
                        ylabel, title):
        """Plot SAM / Anneal / bs512 lines + AdamW baseline on *ax*."""
        valid = [(l, v) for l, v in zip(rho_labels, sam_vals) if v is not None]
        if valid:
            xl, yl = zip(*valid)
            ax.plot(xl, yl, marker="o", color=COLOR_MAP["sam"], label="SAM",
                    alpha=ALPHA, linewidth=2)
        valid_a = [(l, v) for l, v in zip(anneal_labels, anneal_vals)
                   if v is not None]
        if valid_a:
            xl_a, yl_a = zip(*valid_a)
            ax.plot(xl_a, yl_a, marker="s", color="forestgreen",
                    label="SAM Anneal", alpha=ALPHA, linewidth=2, linestyle="--")
        valid_512 = [(l, v) for l, v in zip(rho_labels, bs512_vals)
                     if v is not None]
        if valid_512:
            xl_512, yl_512 = zip(*valid_512)
            ax.plot(xl_512, yl_512, marker="D", color="crimson",
                    label="SAM bs512", alpha=ALPHA, linewidth=2, linestyle="-.")
        if adamw_val is not None:
            ax.axhline(adamw_val, color=COLOR_MAP["adamw"], linestyle=":",
                       linewidth=2, label="AdamW")
        all_vals = [v for v in sam_vals + anneal_vals + bs512_vals
                    + [adamw_val] if v is not None]
        if all_vals:
            span = max(all_vals) - min(all_vals)
            pad = max(span * 0.05, 0.2)
            ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)
        ax.set_xlabel("SAM rho", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel(ylabel, fontsize=FONTSIZE["AXIS"])
        ax.set_title(title, fontsize=FONTSIZE["TITLE"])
        ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"] - 1)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=GRID_ALPHA)

    # ── Figure 1: Overall 4-bit Quantized Average vs rho ──
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    _plot_rho_lines(ax1, sam_q_avgs, anneal_q_avgs, sam_bs512_q_avgs,
                    adamw_q_avg, "4-bit Avg Score",
                    "4-bit Quantized Avg vs SAM rho")
    ax1.legend(fontsize=FONTSIZE["LEGEND"] - 1, loc="upper center",
               bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "5B_ablation_avg.pdf"),
                 bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved 5B ablation avg plot to {out_dir}/5B_ablation_avg.pdf")

    # ── Figure 2: Per-group 4-bit score vs rho (1×3 subplots) ──
    group_names = list(BASE_GROUP_TASKS.keys())
    q_kw = dict(run_type="quant", quant_bit=4, midtrain_tokens=5)

    fig2, axes = plt.subplots(1, len(group_names), figsize=(18, 5))
    for idx, gname in enumerate(group_names):
        ax = axes[idx]
        adamw_g = _group_score(gname, optim="adamw", batch_size=bs, **q_kw)
        sam_g = [_group_score(gname, optim="sam", rho=r, batch_size=bs, **q_kw)
                 for r in rho_values]
        anneal_g = [_group_score(gname, optim="sam", rho=r, batch_size=bs,
                                 anneal_sam=True, **q_kw)
                    for r in anneal_rhos]
        bs512_g = [_group_score(gname, optim="sam", rho=r, batch_size=512, **q_kw)
                   for r in rho_values]

        _plot_rho_lines(ax, sam_g, anneal_g, bs512_g, adamw_g,
                        f"4-bit {gname} Avg", gname)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig2.legend(handles, labels, fontsize=FONTSIZE["LEGEND"] - 1,
                loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=4)
    fig2.suptitle("Per-Group 4-bit Score vs SAM rho",
                  fontsize=FONTSIZE["TITLE"] + 2, y=1.02)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "5B_ablation_per_group.pdf"),
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved 5B per-group plot to {out_dir}/5B_ablation_per_group.pdf")


def main():
    results = parse_results()
    build_base_excel(results)
    build_base_excel_5B(results)
    build_sft_excel(results)
    build_sft_pareto_50B(results)
    build_5B_ablation_summary(results)


if __name__ == "__main__":
    main()
