import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

from utils.helper import get_run_info
from utils.config_globals import ANNEAL_CONFIG, RESULTS_DIR, SIZE, OPTIM, TOKEN_LIST, LRS
from utils.plotting_globals import FIG_WIDTH, OPTIM_MAP, FONTSIZE, COLOR_MAP, LRS_MAP
import json

def pretrain_optim_token(results):

    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)

    for size in SIZE:
        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, 4))

        for optim in OPTIM:
            run_info = get_run_info(results, size, optim)
            if run_info is None:
                print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]}")
                continue 

            x = [r["multiplier"] for r in run_info["pretrain"].values()]
            y = [r["dclm_val"] for r in run_info["pretrain"].values()]

            ax.plot(
                x, y,
                marker="o",
                label=OPTIM_MAP[optim],
                color=COLOR_MAP[optim]
            )

        ax.set_xlabel("Tokens / Param", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Pretrain Loss", fontsize=FONTSIZE["AXIS"])
        ax.set_xscale('log')
        # ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
        ax.legend(fontsize=FONTSIZE["LEGEND"])

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/optim_{size}M_token.png"), bbox_inches='tight')
        plt.close()

def pretrain_optim_compute(results):

    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)

    for size in SIZE:
        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, 4))

        for optim in OPTIM:
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
                marker="o",
                label=OPTIM_MAP[optim],
                color=COLOR_MAP[optim]
            )

        ax.set_xlabel("Relative FLOPs", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Pretrain Loss", fontsize=FONTSIZE["AXIS"])
        ax.set_xscale('log')
        # ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
        ax.legend(fontsize=FONTSIZE["LEGEND"])

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/optim_{size}M_compute.png"), bbox_inches='tight')
        plt.close()



def pretrain_lrs(results, anneal_match="token"):

    os.makedirs(os.path.join(RESULTS_DIR, "plots/pretrain"), exist_ok=True)
    optim = "adamw"

    for size in SIZE:
        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, 4))

        for lrs in LRS:
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
                marker="o",
                label=LRS_MAP[lrs],
                color=COLOR_MAP[lrs]
            )            

        ax.set_xlabel("Tokens / Param", fontsize=FONTSIZE["AXIS"])
        ax.set_ylabel("Pretrain Loss", fontsize=FONTSIZE["AXIS"])
        ax.set_xscale('log')
        # ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
        ax.legend(fontsize=FONTSIZE["LEGEND"])

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plots/pretrain/lrs_{size}M_{anneal_match}.png"), bbox_inches='tight')
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