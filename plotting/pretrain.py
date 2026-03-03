import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import os
import argparse
import numpy as np

from utils.helper import get_run_info
from utils.config_globals import ANNEAL_CONFIG, RESULTS_DIR, SIZE, OPTIM, TOKEN_LIST, LRS, ALL_LRS
from utils.plotting_globals import FIG_WIDTH, FIG_HEIGHT, GRID_ALPHA, OPTIM_MAP, FONTSIZE, COLOR_MAP, LRS_MAP, ALPHA, XLABEL, YLABEL, LEGEND_PARAM, MARKERS
import json


# def pretrain_all_token(results):

#     os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)

#     fig, axs = plt.subplots(1, len(SIZE), figsize=(FIG_WIDTH * 2, FIG_HEIGHT), sharex=True)
#     for col, size in enumerate(SIZE):
#         ax = axs[col]
        
#         for config in ALL_LRS:
#             optim = config.split("_")[1]
#             if "cosine" in config:
#                 run_info = get_run_info(results, size, optim, pt_lr="adapt")
#             elif "wsd" in config:
#                 if optim == "sam":
#                     run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=optim, anneal_match="token")
#                 else:
#                     run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=optim, anneal_match="token")
#             else:
#                 raise ValueError(f"Invalid config: {config}")

#             if run_info is None:
#                 print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]}")
#                 continue 

#             x = [r["multiplier"] for r in run_info["pretrain"].values()]
#             y = [r["dclm_val"] for r in run_info["pretrain"].values()]

#             ax.plot(
#                 x, y,
#                 marker="o",
#                 label=LRS_MAP[config],
#                 color=COLOR_MAP[config],
#                 alpha=ALPHA
#             )

#         ax.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
#         if col == 0:
#             ax.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
#         ax.set_xscale('log')
#         ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
#         ax.grid(True)
#         ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
#         ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])

#     handles, labels = axs[len(SIZE)-1].get_legend_handles_labels()
#     fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])

#     plt.tight_layout()
#     plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/all_token.pdf"), bbox_inches='tight')
#     plt.close()


def pretrain_optim_token(results):

    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)
    fig, axs = plt.subplots(1, len(SIZE), figsize=(FIG_WIDTH, 2.25), sharex=True)
    for col, size in enumerate(SIZE):
        ax = axs[col]
        
        for idx, optim in enumerate(OPTIM):
            run_info = get_run_info(results, size, optim)
            if run_info is None:
                print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]}")
                continue 

            x = [r["multiplier"] for r in run_info["pretrain"].values()]
            y = [r["dclm_val"] for r in run_info["pretrain"].values()]

            ax.plot(
                x, y,
                label=OPTIM_MAP[optim],
                color=COLOR_MAP[optim],
                alpha=ALPHA,
                marker=MARKERS[idx]
            )

        ax.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
        if col == 0:
            ax.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
        ax.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
        ax.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
        ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
    

    handles, labels = axs[len(SIZE)-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/optim_token.pdf"), bbox_inches='tight')
    plt.close()



def pretrain_optim_compute(results):

    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)
    fig, axs = plt.subplots(1, len(SIZE), figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
    for col, size in enumerate(SIZE):
        ax = axs[col]

        for idx, optim in enumerate(OPTIM):
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

            ax.plot(
                rel_flops, y,
                label=OPTIM_MAP[optim],
                color=COLOR_MAP[optim],
                alpha=ALPHA,
                marker=MARKERS[idx]
            )

        ax.set_xlabel(XLABEL["RELATIVE_FLOPS"], fontsize=FONTSIZE["AXIS"])
        if col == 0:
            ax.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
        ax.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
        ax.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)

        ax.set_xlim(0, 10)
        ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
    
    handles, labels = axs[len(SIZE)-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/optim_compute.pdf"), bbox_inches='tight')
    plt.close()



def pretrain_lrs(results, anneal_match="token"):

    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)
    optim = "adamw"

    fig, axs = plt.subplots(1, len(SIZE), figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
    for col, size in enumerate(SIZE):
        ax = axs[col]

        for idx, lrs in enumerate(LRS):
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
        
            ax.plot(
                x, y,
                label=LRS_MAP[lrs],
                color=COLOR_MAP[lrs],
                alpha=ALPHA,
                marker=MARKERS[idx]
            )            

        ax.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
        if col == 0:
            ax.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
        ax.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
        ax.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
        ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
    
    handles, labels =  axs[len(SIZE)-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/lrs_{anneal_match}.pdf"), bbox_inches='tight')
    plt.close()


def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    if args.plot == "optim":
        if args.type == "token":
            pretrain_optim_token(results)
        elif args.type == "compute":
            pretrain_optim_compute(results)        
    elif args.plot == "lrs":
        if args.type == "token":
            pretrain_lrs(results, anneal_match="token")
        elif args.type == "compute":
            raise NotImplementedError("Compute plots not implemented yet")
    elif args.plot == "all":
        pretrain_optim_token(results)
        pretrain_optim_compute(results)
        pretrain_lrs(results, anneal_match="token")
    else:
        raise ValueError(f"Invalid plot: {args.plot}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pretrain Plotting Tool")
    parser.add_argument(
        "--plot",
        type=str,
        default="all",
        choices=["optim", "lrs", "all"],
        help="Which plot to generate: optim, lrs, all",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="token",
        choices=["token", "compute"],
        help="Which type of plot to generate: token, compute",
    )
    args = parser.parse_args()
    main(args)