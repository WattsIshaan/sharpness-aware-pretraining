import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse

from utils.config_globals import PERTURBATIONS, PT_LR, SIZE, OPTIM, TOKEN_LIST, RESULTS_DIR, ANNEAL_CONFIG, LRS, PERTURBATION_THRESHOLD
from utils.plotting_globals import OPTIM_MAP, FONTSIZE, FIG_WIDTH, FIG_HEIGHT, COLOR_MAP, LRS_MAP, LEGEND_PARAM, ALPHA, GRID_ALPHA, MARKERS
from utils.helper import get_run_info
from utils.plotting_globals import XLABEL, YLABEL
from matplotlib.ticker import NullLocator



def perturbation_optim(results):

    for size in SIZE:
        perturbations = [p for p in PERTURBATIONS if p <= PERTURBATION_THRESHOLD]
        fig, axs = plt.subplots(1, len(OPTIM), figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True, sharex=True)
        cmap = plt.get_cmap('viridis', len(perturbations))

        for idx, optim in enumerate(OPTIM):
            run_info = get_run_info(results, size, optim, perturb=True)
            if run_info == None:
                continue
            ax = axs[idx]
            for i, perturbation in enumerate(perturbations):
                
                x = [r["multiplier"] for r in run_info["perturbed"][perturbation].values()]  
                y = [r["dclm_perturbed"] for r in run_info["perturbed"][perturbation].values()]
                ax.plot(x, y, label=f"std = {perturbation}", marker='o', color=cmap(i), alpha=ALPHA)

            ax.set_title(f"{OPTIM_MAP[optim]}")
            ax.grid()

            ax.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
            ax.set_xscale('log')
            ax.set_ylabel(YLABEL["PT_LOSS_PERTURBED"], fontsize=FONTSIZE["AXIS"])
            ax.minorticks_on()
            ax.yaxis.set_minor_locator(NullLocator())
            ax.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
            ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])

            if size == 150:
                ax.set_xlim(ax.get_xlim()[0], 1010)
            
        # Use only one legend for the entire figure; use the handles/labels from the first axis
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles))
            
        plt.tight_layout()  # type: ignore
        
        os.makedirs(os.path.join(RESULTS_DIR, f"plots/perturbation/optim/"), exist_ok=True)
        plt.savefig(f"results/plots/perturbation/optim/{size}m.pdf", bbox_inches="tight")
        plt.close()


def perturbation_lrs(results):
 
    for size in SIZE:
        perturbations = [p for p in PERTURBATIONS if p <= PERTURBATION_THRESHOLD]
        fig, axs = plt.subplots(1, len(LRS), figsize=(1.5 * FIG_WIDTH, FIG_HEIGHT), sharey=True, sharex=True)
        cmap = plt.get_cmap('viridis', len(perturbations))

        for col, lrs in enumerate(LRS):
            if lrs == "cosine":
                run_info = get_run_info(results, size, 'adamw', perturb=True)
            else:
                anneal_optim = lrs.split("_")[1]
                run_info = get_run_info(results, size, 'adamw', anneal=True, perturb=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token")

            if run_info == None:
                print(f"Skipping {lrs} for OLMo {size}M")
                continue
            ax = axs[col]
            for i, perturbation in enumerate(perturbations):
                x = None
                x = [r["multiplier"] for r in run_info["perturbed"][perturbation].values()]
                y = [r["dclm_perturbed"] for r in run_info["perturbed"][perturbation].values()]
                ax.plot(x, y, label=f"std = {perturbation}", marker='o', color=cmap(i), alpha=ALPHA)

            ax.set_title(f"{LRS_MAP[lrs]}")
            ax.grid()

            ax.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
            ax.set_xscale('log')
            if col == 0:
                ax.set_ylabel(YLABEL["PT_LOSS_PERTURBED"], fontsize=FONTSIZE["AXIS"])
            
            ax.minorticks_on()
            ax.yaxis.set_minor_locator(NullLocator())
            ax.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
            ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])

            if size == 150:
                ax.set_xlim(ax.get_xlim()[0], 1010)
            
        # Use only one legend for the entire figure; use the handles/labels from the first axis
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles))
            
        plt.tight_layout()  # type: ignore
        
        os.makedirs(os.path.join(RESULTS_DIR, f"plots/perturbation/lrs/"), exist_ok=True)
        plt.savefig(f"results/plots/perturbation/lrs/{size}m.pdf", bbox_inches="tight")
        plt.close()






def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    # perturbation_sam_vs_adamw_cosine(results)
    if args.plot == "lrs":
        perturbation_lrs(results)
    elif args.plot == "optim":
        perturbation_optim(results)
    elif args.plot == "all":
        perturbation_optim(results)
        perturbation_lrs(results)
    else:
        raise ValueError(f"Invalid plot: {args.plot}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perturbation Plotting Tool")
    parser.add_argument(
        "--plot",
        type=str,
        default="all",
        choices=["optim", "lrs", "all"],
        help="Which plot to generate: optim, lrs, all",
    )
    args = parser.parse_args()
    main(args)