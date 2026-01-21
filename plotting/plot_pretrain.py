import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

from utils.helper import get_run_info
from utils.config_globals import ANNEAL_CONFIG, RESULTS_DIR, SIZE, OPTIM, TOKEN_LIST
from utils.plotting_globals import FIG_WIDTH, OPTIM_MAP, FONTSIZE, COLOR_MAP
import json

def make_fixed_token_plot(results):

    n_sizes = len(SIZE)
    fig, axs = plt.subplots(1, n_sizes, figsize=(FIG_WIDTH * n_sizes, 4), sharex=True)

    # To handle the case where axs is not iterable if n_sizes == 1
    if n_sizes == 1:
        axs = [axs]

    plot_handles = []
    plot_labels = []

    for i, (size, ax) in enumerate(zip(SIZE, axs)):
        for j, optim in enumerate(OPTIM):
            
                run_info = get_run_info(results, size, optim)
                if run_info is None:
                    print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]}")
                    continue 

                x = [r["multiplier"] for r in run_info["pretrain"].values()]
                y = [r["dclm_val"] for r in run_info["pretrain"].values()]

                handle, = ax.plot(
                    x, y,
                    marker="o",
                    label=OPTIM_MAP[optim],
                    color=COLOR_MAP[optim]
                )

                if i == 0:
                    plot_handles.append(handle)
                    plot_labels.append(f"{OPTIM_MAP[optim]}")
            

        ax.set_xlabel("Tokens / Param", fontsize=FONTSIZE["AXIS"])
        if i == 0:
            ax.set_ylabel("DCLM Val Loss", fontsize=FONTSIZE["AXIS"])
        ax.set_xscale('log')
        ax.set_title(f"OLMo-{size}M", fontsize=FONTSIZE["TITLE"])
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])

    # Only show legend on the last axis (or can do fig-wide)
    axs[2].legend(plot_handles, plot_labels, fontsize=FONTSIZE["LEGEND"])
    plt.suptitle(f"Pretrain Val Loss (Token Matched)", fontsize=FONTSIZE["SUP_TITLE"])

    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/optim_all_size_token.png"), bbox_inches='tight')
    plt.close()

def make_fixed_compute_plot(results):

    n_sizes = len(SIZE)
    fig, axs = plt.subplots(1, n_sizes, figsize=(FIG_WIDTH * n_sizes, 4), sharex=True)

    # To handle the case where axs is not iterable if n_sizes == 1
    if n_sizes == 1:
        axs = [axs]

    plot_handles = []
    plot_labels = []

    for i, (size, ax) in enumerate(zip(SIZE, axs)):
        for j, optim in enumerate(OPTIM):
            
                run_info = get_run_info(results, size, optim)
                if run_info is None:
                    print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]}")
                    continue 

                x = [r["multiplier"] for r in run_info["pretrain"].values()]
                min_multiplier = min(x)
                rel_flops = [f/min_multiplier for f in x][:-1]
                y = [r["dclm_val"] for r in run_info["pretrain"].values()]

                if optim == "sam":
                    y = y[:-1]
                elif optim == "adamw":
                    y = y[1:]

                handle, = ax.plot(
                    rel_flops, y,
                    marker="o",
                    label=OPTIM_MAP[optim],
                    color=COLOR_MAP[optim]
                )

                if i == 0:
                    plot_handles.append(handle)
                    plot_labels.append(f"{OPTIM_MAP[optim]}")
            

        ax.set_xlabel("Relative FLOPs", fontsize=FONTSIZE["AXIS"])
        if i == 0:
            ax.set_ylabel("DCLM Val Loss", fontsize=FONTSIZE["AXIS"])
        ax.set_xscale('log')
        ax.set_title(f"OLMo-{size}M", fontsize=FONTSIZE["TITLE"])
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])

    # Only show legend on the last axis (or can do fig-wide)
    axs[2].legend(plot_handles, plot_labels, fontsize=FONTSIZE["LEGEND"])
    plt.suptitle(f"Pretrain Val Loss (Compute Matched)", fontsize=FONTSIZE["SUP_TITLE"])

    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/optim_all_size_compute.png"), bbox_inches='tight')
    plt.close()


def make_lrs_plot(results):


    n_sizes = len(SIZE)
    fig, axs = plt.subplots(1, n_sizes, figsize=(FIG_WIDTH * n_sizes, 4), sharex=True)
    optim = "adamw"

    for col, size in enumerate(SIZE):
        token_budget = TOKEN_LIST[size][-1]
        ax = axs[col]
        for pt_lrs in ["cosine", "wsd"]:
            if pt_lrs == "cosine":
                run_info = get_run_info(results, size, optim, anneal=False)

                if run_info is None:
                    continue

                cosine_dclm_val = run_info["pretrain"][token_budget]["dclm_val"]
                ax.axhline(
                    cosine_dclm_val,
                    label="Cosine",
                    color="black",
                    linestyle="--"
                )

            elif pt_lrs == "wsd":

                anneal_percent = ANNEAL_CONFIG["percent"]
                for anneal_optim in ANNEAL_CONFIG["anneal_optim"]:
                    anneal_dclm_val = np.empty_like(anneal_percent, dtype=float)
                    for j, p in enumerate(anneal_percent):
                        run_info = get_run_info(results, size, optim, anneal=True, anneal_percent=p, anneal_optim=anneal_optim)

                        if run_info is None:
                            anneal_dclm_val[j] = np.nan # type: ignore
                            continue
                        
                        anneal_dclm_val[j] = run_info["pretrain"][token_budget]["dclm_val"]

                    ax.plot(
                        anneal_percent,
                        anneal_dclm_val,
                        color=COLOR_MAP[anneal_optim], 
                        marker="o",
                        label=f"WSD (Anneal Optim: {OPTIM_MAP[anneal_optim]})"
                    )

        ax.set_xlabel("Percentage of Total Steps for Annealing", fontsize=FONTSIZE["AXIS"])
        if col == 0:
            ax.set_ylabel(f"Pretrain DCLM Val Loss", fontsize=FONTSIZE["AXIS"])
        ax.set_title(f"OLMo-{size}M", fontsize=FONTSIZE["TITLE"])

        if col == len(SIZE)-1:
            ax.legend(fontsize=FONTSIZE["LEGEND"])

        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
        ax.grid(True)

    fig.subplots_adjust(top=0.83)  # Make room for the suptitle before tight_layout
    fig.suptitle(f"Pretrain DCLM Val Loss across Learning Rate Schedulers", fontsize=FONTSIZE["SUP_TITLE"])
    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)
    # plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/lrs_all_size_token.png"), bbox_inches='tight')
    plt.close()


def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    if args.plot == "optim_token":
        make_fixed_token_plot(results)
    elif args.plot == "optim_compute":
        make_fixed_compute_plot(results)
    elif args.plot == "lrs":
        make_lrs_plot(results)
    
    else:
        make_fixed_compute_plot(results)
        make_fixed_token_plot(results)
        make_lrs_plot(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pretrain Plotting Tool")
    parser.add_argument(
        "--plot",
        type=str,
        choices=["all", "optim_token", "optim_compute", "lrs"],
        default="all",
        help="Which plot to generate: all, fixed_token, fixed_compute, lrs (default: all)",
    )

    args = parser.parse_args()
    main(args)