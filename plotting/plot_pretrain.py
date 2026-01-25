import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

from utils.helper import get_run_info
from utils.config_globals import ANNEAL_CONFIG, RESULTS_DIR, SIZE, OPTIM, TOKEN_LIST, LRS
from utils.plotting_globals import FIG_WIDTH, OPTIM_MAP, FONTSIZE, COLOR_MAP, LRS_MAP
import json

def pretrain_optim_token(results):

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

def pretrain_optim_compute(results):

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



def pretrain_lrs(results, anneal_match="token"):

    n_sizes = len(SIZE)
    fig, axs = plt.subplots(1, n_sizes, figsize=(FIG_WIDTH * n_sizes, 4), sharex=True)
    optim = "adamw"

    # To handle the case where axs is not iterable if n_sizes == 1
    if n_sizes == 1:
        axs = [axs]

    plot_handles = []
    plot_labels = []

    for col, (size, ax) in enumerate(zip(SIZE, axs)):
        for lrs in (LRS):

            if lrs == "cosine":
                    run_info = get_run_info(results, size, optim, anneal=False)
            else:
                anneal_optim = lrs.split("_")[1]
                if optim == "sam":
                    run_info = get_run_info(results, size, optim, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match=anneal_match)
                else:
                    run_info = get_run_info(results, size, optim, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token")

            if run_info is None:
                print(f"Skipping {lrs} for OLMo {size}M")
                continue

            x = [r["multiplier"] for r in run_info["pretrain"].values()]
            y = [r["dclm_val"] for r in run_info["pretrain"].values()]
        
            handle, = ax.plot(
                x, y,
                marker="o",
                label=LRS_MAP[lrs],
                color=COLOR_MAP[lrs]
            )

            if col == 1:
                plot_handles.append(handle)
                plot_labels.append(f"{LRS_MAP[lrs]}")
            

        ax.set_xlabel("Tokens / Param", fontsize=FONTSIZE["AXIS"])
        if col == 0:
            ax.set_ylabel("DCLM Val Loss", fontsize=FONTSIZE["AXIS"])
        ax.set_xscale('log')
        ax.set_title(f"OLMo-{size}M", fontsize=FONTSIZE["TITLE"])
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])

    # Only show legend on the last axis (or can do fig-wide)
    axs[2].legend(plot_handles, plot_labels, fontsize=FONTSIZE["LEGEND"])
    plt.suptitle(f"Pretrain Val Loss across Learning Rate Schedulers ({anneal_match.capitalize()} Matched)", fontsize=FONTSIZE["SUP_TITLE"])

    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/lrs_all_size_{anneal_match}.png"), bbox_inches='tight')
    plt.close()






def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    if args.plot == "optim_token":
        pretrain_optim_token(results)
    elif args.plot == "optim_compute":
        pretrain_optim_compute(results)
    elif args.plot == "lrs_token":
        pretrain_lrs(results, anneal_match="token")
    elif args.plot == "lrs_compute":
        pretrain_lrs(results, anneal_match="compute")
    else:
        pretrain_optim_compute(results)
        pretrain_optim_token(results)
        pretrain_lrs(results, anneal_match="token")
        pretrain_lrs(results, anneal_match="compute")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pretrain Plotting Tool")
    parser.add_argument(
        "--plot",
        type=str,
        choices=["all", "optim_token", "optim_compute", "lrs_token", "lrs_compute"],
        default="all",
        help="Which plot to generate: all, fixed_token, fixed_compute, lrs (default: all)",
    )

    args = parser.parse_args()
    main(args)