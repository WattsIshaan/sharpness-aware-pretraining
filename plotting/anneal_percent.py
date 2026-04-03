import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.spatial import ConvexHull

from utils.config_globals import RESULTS_DIR, PERTURBATIONS, TOKEN_LIST
from utils.plotting_globals import (
    FONTSIZE,
    FIG_WIDTH,
    FIG_HEIGHT,
    LEGEND_PARAM,
    ALPHA,
    GRID_ALPHA,
    MARKERS,
    XLABEL,
    YLABEL,
    CPT_DATASET_MAP,
)
from utils.helper import get_run_info
from pareto import YLIM, XLIM

ANNEAL_PERCENTS = [5, 10, 20, 50, 100]
SIZE = 60
TOKEN = TOKEN_LIST[SIZE][-1]  # 192

# Perturbation plot: cap y-axis here; series are clipped at the axes patch (default clip_on).
PERTURB_PLOT_Y_MAX = 5.0

PERCENT_COLORS = {
    5: "grey",
    10: "forestgreen",
    20: "violet",
    50: "royalblue",
    100: "darkorange",
}
PERCENT_LABELS = {p: f"{p}%" for p in ANNEAL_PERCENTS}


def _pareto_style_hull_xy(dclm_val, cpt_val):
    """
    Convex hull boundary segment used in ``pareto.py`` (``pareto_optim_size``):
    ``ConvexHull`` → rotate by min-x → truncate when x stops increasing.
    Returns ``(N, 2)`` array or ``None`` if fewer than 3 points.
    """
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


def _cpt_lr_label(lr):
    """Format ``cpt_lr`` for point labels (same compact style as ``pareto_1b._lr_label``)."""
    if lr is None or lr == "unknown":
        return ""
    try:
        f = float(lr)
    except (TypeError, ValueError):
        return ""
    s = f"{f:.2e}".replace("+", "")
    c, e = s.split("e")
    c = c.rstrip("0").rstrip(".")
    return f"{c}e{e}"


def _cpt_pareto_series_like_pareto(run_info, cpt_dataset, token_budget, size):
    """
    Same CPT extraction as ``pareto.pareto_optim_size`` (lines 90–112): ``get_run_info`` nest,
    ``cpt_lrs = sorted(run_info['cpt'].keys())``, then nested loops over sorted ``cpt_wd`` / ``cpt_bs``,
    ``XLIM``/``YLIM`` gate, and Starcoder ``x_val > 5.1`` skip when ``size == 20``.
    """
    if not run_info or "cpt" not in run_info:
        return [], [], []
    dclm_val = []
    cpt_val = []
    lr_vals = []
    cpt_lrs = sorted(run_info["cpt"].keys())
    for cpt_lr in cpt_lrs:
        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
        for cpt_wd in cpt_wds:
            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
            for cpt_bs in cpt_bss:
                try:
                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget]["dclm_val"]
                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget][cpt_dataset]
                    if (
                        x_val >= XLIM[size][cpt_dataset][0]
                        and x_val <= XLIM[size][cpt_dataset][1]
                        and y_val >= YLIM[size][cpt_dataset][0]
                        and y_val <= YLIM[size][cpt_dataset][1]
                    ):
                        if size == 20 and cpt_dataset == "starcoder":
                            if x_val > 5.1:
                                continue
                        dclm_val.append(x_val)
                        cpt_val.append(y_val)
                        lr_vals.append(cpt_lr)
                    else:
                        print("Not in window")
                except Exception as e:
                    print("Exception", e)
                    continue
    # print(len(dclm_val))
    return dclm_val, cpt_val, lr_vals


# ── 2.1  Quantisation bar plot ──────────────────────────────────────────────

def plot_quant_bar(results):
    """Bar plot: 5 bars for anneal percentages, y = 4-bit quantised dclm_val."""

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.7, FIG_HEIGHT * 1.2))

    x = np.arange(len(ANNEAL_PERCENTS))
    bar_width = 0.55
    vals = []
    colors = []

    for ap in ANNEAL_PERCENTS:
        run_info = get_run_info(
            results, SIZE, "adamw",
            quantized=True, model_type="hf",
            anneal=True, anneal_percent=ap, anneal_optim="adamw",
            pretrain_lrs="wsd2",
        )
        if run_info and run_info["quantized"].get(4, {}).get(TOKEN):
            vals.append(run_info["quantized"][4][TOKEN]["dclm_quant"])
        else:
            vals.append(None)
        colors.append(PERCENT_COLORS[ap])

    valid_vals = [v for v in vals if v is not None]

    ax.bar(
        x, vals, bar_width,
        color=colors,
        alpha=ALPHA,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([PERCENT_LABELS[p] for p in ANNEAL_PERCENTS], fontsize=FONTSIZE["TICKS"])
    ax.set_xlabel("Anneal Percentage", fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(YLABEL["PT_LOSS_QUANTIZED"], fontsize=FONTSIZE["AXIS"])
    ax.tick_params(axis="y", labelsize=FONTSIZE["TICKS"])
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_title("4-bit Quantisation", fontsize=FONTSIZE["TITLE"])

    # ylim ±5% of min/max
    if valid_vals:
        y_min, y_max = min(valid_vals), max(valid_vals)
        ax.set_ylim(y_min * 0.95, y_max * 1.05)

    plt.tight_layout()
    out_dir = os.path.join(RESULTS_DIR, "plots/anneal_percent")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "quant_bar.png"), bbox_inches="tight")
    plt.close()
    print("Saved quant_bar.png")


# ── 2.1b  Pretrain loss after annealing (bar, 60M OLMo, 192B) ────────────────

def plot_pretrain_loss_after_anneal(results):
    """
    Bar plot: DCLM pretrain validation loss at 192B tokens vs anneal fraction
    (5–100%), WSD2 + AdamW anneal, 60M ``model_type=olmo`` (via ``get_run_info``).
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.7, FIG_HEIGHT * 1.2))

    x = np.arange(len(ANNEAL_PERCENTS))
    bar_width = 0.55
    vals = []
    colors = []

    for ap in ANNEAL_PERCENTS:
        run_info = get_run_info(
            results,
            SIZE,
            "adamw",
            anneal=True,
            anneal_percent=ap,
            anneal_optim="adamw",
            pretrain_lrs="wsd2",
        )
        entry = run_info["pretrain"].get(TOKEN) if run_info else None
        if entry and entry.get("dclm_val") is not None:
            vals.append(entry["dclm_val"])
        else:
            vals.append(None)
        colors.append(PERCENT_COLORS[ap])

    valid_vals = [v for v in vals if v is not None]

    ax.bar(
        x,
        vals,
        bar_width,
        color=colors,
        alpha=ALPHA,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([PERCENT_LABELS[p] for p in ANNEAL_PERCENTS], fontsize=FONTSIZE["TICKS"])
    ax.set_xlabel("Anneal percentage", fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
    ax.tick_params(axis="y", labelsize=FONTSIZE["TICKS"])
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_title(
        f"{SIZE}M OLMo · {TOKEN}B PT Tokens",
        fontsize=FONTSIZE["TITLE"],
    )

    if valid_vals:
        y_min, y_max = min(valid_vals), max(valid_vals)
        pad = (y_max - y_min) * 0.08 + 1e-6
        ax.set_ylim(y_min - pad, y_max + pad)

    plt.tight_layout()
    out_dir = os.path.join(RESULTS_DIR, "plots/anneal_percent")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "pretrain_loss.png"), bbox_inches="tight")
    plt.close()
    print("Saved pretrain_anneal_bar.png")


# ── 2.2  Pareto (StarCoder only, 60M 192B) — same hull/CPT extraction as ``ewc_cpt`` / ``pareto`` ──

PARETO_CPT_DATASET = "starcoder"

# Cosine pretrain schedule (contrasted with WSD2 + anneal %); matches ``preprocess_results`` (no ``anneal_percent``).
COSINE_PARETO_COLOR = "red"
COSINE_PARETO_MARKER = "^"


def plot_pareto(results):
    """One panel: DCLM vs StarCoder CPT loss at 60M / 192B. WSD2 (anneal %) + cosine (``ewc_cpt``-style axes)."""
    fig, ax = plt.subplots(figsize=(4, FIG_HEIGHT*1.2))
    cpt_tokens = 10
    cpt_dataset = PARETO_CPT_DATASET

    lr_fs = max(FONTSIZE["TICKS"] - 4, 6)
    has_cosine_any = False

    for i, ap in enumerate(ANNEAL_PERCENTS):
        run_info_wsd = get_run_info(
            results,
            SIZE,
            "adamw",
            cpt_dataset,
            cpt_tokens=cpt_tokens,
            anneal=True,
            anneal_percent=ap,
            anneal_optim="adamw",
            pretrain_lrs="wsd2",
        )
        dclm_vals, cpt_vals, lr_vals = _cpt_pareto_series_like_pareto(
            run_info_wsd, cpt_dataset, TOKEN, SIZE
        )
        if not dclm_vals:
            continue
        color = PERCENT_COLORS[ap]
        ax.scatter(
            dclm_vals,
            cpt_vals,
            color=color,
            marker=MARKERS[i % len(MARKERS)],
            alpha=ALPHA,
        )
        if len(dclm_vals) > 2:
            hull_xy = _pareto_style_hull_xy(dclm_vals, cpt_vals)
            if hull_xy is not None:
                ax.plot(
                    hull_xy[:, 0],
                    hull_xy[:, 1],
                    color=PERCENT_COLORS[ap],
                    label=PERCENT_LABELS[ap],
                    marker=MARKERS[i % len(MARKERS)],
                    alpha=ALPHA,
                )
        else:
            ax.plot(
                [],
                [],
                color=PERCENT_COLORS[ap],
                label=PERCENT_LABELS[ap],
                marker=MARKERS[i % len(MARKERS)],
            )

    run_info_cos = get_run_info(
        results,
        SIZE,
        "adamw",
        cpt_dataset,
        cpt_tokens=cpt_tokens,
        anneal=False,
        pretrain_lrs="cosine",
    )
    dclm_cos, cpt_cos, lr_cos = _cpt_pareto_series_like_pareto(
        run_info_cos, cpt_dataset, TOKEN, SIZE
    )

    if dclm_cos:
        has_cosine_any = True
        ax.scatter(
            dclm_cos,
            cpt_cos,
            color=COSINE_PARETO_COLOR,
            marker=COSINE_PARETO_MARKER,
            alpha=ALPHA,
            zorder=4,
        )
        for xv, yv, lr_v in zip(dclm_cos, cpt_cos, lr_cos):
            lab = _cpt_lr_label(lr_v)
            # if lab:
            #     ax.annotate(
            #         lab,
            #         (xv, yv),
            #         xytext=(3, 3),
            #         textcoords="offset points",
            #         fontsize=lr_fs,
            #         color=COSINE_PARETO_COLOR,
            #         alpha=0.95,
            #         zorder=5,
            #     )
        if len(dclm_cos) > 2:
            hull_xy_c = _pareto_style_hull_xy(dclm_cos, cpt_cos)
            if hull_xy_c is not None:
                ax.plot(
                    hull_xy_c[:, 0],
                    hull_xy_c[:, 1],
                    color=COSINE_PARETO_COLOR,
                    marker=COSINE_PARETO_MARKER,
                    alpha=ALPHA,
                    zorder=3,
                )

    ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
    ax.set_title(
        f"{SIZE}M · {CPT_DATASET_MAP[cpt_dataset]} · {TOKEN}B",
        fontsize=FONTSIZE["TITLE"],
    )
    ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"])
    ax.grid(True, alpha=GRID_ALPHA)

    # ax.set_xlim(XLIM[SIZE][cpt_dataset][0], XLIM[SIZE][cpt_dataset][1])
    # ax.set_ylim(YLIM[SIZE][cpt_dataset][0], YLIM[SIZE][cpt_dataset][1])

    if has_cosine_any:
        ax.plot(
            [],
            [],
            color=COSINE_PARETO_COLOR,
            label="Cosine",
            marker=COSINE_PARETO_MARKER,
            alpha=ALPHA,
        )
    handles, labels_leg = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels_leg,
        title="Anneal %" if not has_cosine_any else None,
        # fontsize=FONTSIZE["LEGEND"],
        # title_fontsize=FONTSIZE["LEGEND"],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        ncol=1,
        alignment="left",
    )
    plt.tight_layout(rect=[0.03, 0.06, 0.78, 0.94])
    out_dir = os.path.join(RESULTS_DIR, "plots/anneal_percent")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "pareto.png"), bbox_inches="tight")
    plt.close()
    print("Saved pareto.png")


# ── 2.3  Perturbation line plot ─────────────────────────────────────────────

def plot_perturbation(results):
    """Line plot: dclm_val (y) vs increasing perturbation (x), 5 lines for anneal %."""

    perturbations = sorted(PERTURBATIONS)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.7, FIG_HEIGHT * 1.2))
    y_all: list[float] = []

    for i, ap in enumerate(ANNEAL_PERCENTS):
        run_info = get_run_info(
            results, SIZE, "adamw",
            perturb=True,
            anneal=True, anneal_percent=ap, anneal_optim="adamw",
            pretrain_lrs="wsd2",
        )
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
                label=PERCENT_LABELS[ap],
                marker=MARKERS[i % len(MARKERS)],
                color=PERCENT_COLORS[ap],
                alpha=ALPHA,
            )

    if y_all:
        y_lo = min(y_all)
        pad = max(0.02, (PERTURB_PLOT_Y_MAX - y_lo) * 0.04)
        ax.set_ylim(y_lo - pad, PERTURB_PLOT_Y_MAX)
    else:
        ax.set_ylim(0.0, PERTURB_PLOT_Y_MAX)

    ax.set_xlabel("Perturbation (γ)", fontsize=FONTSIZE["AXIS"])
    ax.set_ylabel(YLABEL["PT_LOSS_PERTURBED"], fontsize=FONTSIZE["AXIS"])
    ax.set_title("Perturbation", fontsize=FONTSIZE["TITLE"])
    ax.tick_params(axis="both", labelsize=FONTSIZE["TICKS"])
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

    # Horizontal legend at bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Anneal %",
        fontsize=FONTSIZE["LEGEND"],
        title_fontsize=FONTSIZE["LEGEND"],
        loc=LEGEND_PARAM["LOC"],
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(handles),
    )

    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.22)
    out_dir = os.path.join(RESULTS_DIR, "plots/anneal_percent")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "perturbation.png"), bbox_inches="tight")
    plt.close()
    print("Saved perturbation.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as f:
        results = json.load(f)

    out_dir = os.path.join(RESULTS_DIR, "plots/anneal_percent")
    os.makedirs(out_dir, exist_ok=True)

    plot_quant_bar(results)
    plot_pretrain_loss_after_anneal(results)
    plot_perturbation(results)
    plot_pareto(results)


if __name__ == "__main__":
    main()
