import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse

from utils.config_globals import PERTURBATIONS, PT_LR, SIZE, OPTIM, TOKEN_LIST, RESULTS_DIR, ANNEAL_CONFIG, LRS
from utils.plotting_globals import OPTIM_MAP, FONTSIZE, FIG_WIDTH, COLOR_MAP, LRS_MAP
from utils.helper import get_run_info

PERTURBATIONS_THRESHOLD = 0.025

def perturbation_sam_vs_adamw_cosine(results):

    for match in ["token", "loss"]:
        for size in SIZE:
            perturbations = [p for p in PERTURBATIONS if p <= PERTURBATIONS_THRESHOLD]
            fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_WIDTH, 4), sharey=True, sharex=True)
            cmap = plt.get_cmap('viridis', len(perturbations))

            for idx, optim in enumerate(OPTIM):
                run_info = get_run_info(results, size, optim, perturb=True)
                if run_info == None:
                    continue
                ax = axs[idx]
                for i, perturbation in enumerate(perturbations):
                    
                    x = None
                    if match == "token":
                        x = [r["multiplier"] for r in run_info["perturbed"][perturbation].values()]
                    elif match == "loss":
                        x = [r["dclm_val"] for r in run_info["perturbed"][perturbation].values()]    
                    y = [r["dclm_perturbed"] for r in run_info["perturbed"][perturbation].values()]
                    ax.plot(x, y, label=f"std = {perturbation}", marker='o', color=cmap(i))

                ax.set_title(f"{OPTIM_MAP[optim]}")
                ax.grid()

                if match == "token":
                    ax.set_xlabel("Tokens / Param", fontsize=FONTSIZE["AXIS"])
                    ax.set_xscale('log')
                else:
                    ax.set_xlabel("Pretrain Loss", fontsize=FONTSIZE["AXIS"])
                if idx == 0:
                    ax.set_ylabel("Pretrain Loss after Perturbation", fontsize=FONTSIZE["AXIS"])

                ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
                
            # Use only one legend for the entire figure; use the handles/labels from the first axis
            handles, labels = axs[0].get_legend_handles_labels()
            axs[1].legend(handles, labels, fontsize=FONTSIZE["LEGEND"])

            # if match == "token":
            #     plt.suptitle(f"OLMo-{size}M DCLM Val Loss after Perturbation vs Pretrain Token / Param (Cosine)", fontsize=FONTSIZE["SUP_TITLE"])
            # elif match == "loss":
            #     plt.suptitle(f"OLMo-{size}M DCLM Val Loss after Perturbation vs DCLM Pretrain Val Loss (Cosine)", fontsize=FONTSIZE["SUP_TITLE"])
                
            plt.tight_layout()  # type: ignore
            
            os.makedirs(os.path.join(RESULTS_DIR, f"plots/perturbation/optim_cosine/"), exist_ok=True)
            plt.savefig(f"results/plots/perturbation/optim_cosine/{size}m_{match}.png", bbox_inches="tight")
            plt.close()


def perturbation_lrs(results):

    for match in ["token", "loss"]:
        for size in SIZE:
            perturbations = [p for p in PERTURBATIONS if p <= PERTURBATIONS_THRESHOLD]
            fig, axs = plt.subplots(1, len(LRS), figsize=(2 * FIG_WIDTH, 4), sharey=True, sharex=True)
            cmap = plt.get_cmap('viridis', len(perturbations))
            optim = "adamw"

            for col, lrs in enumerate(LRS):

                if lrs == "cosine":
                    run_info = get_run_info(results, size, optim, perturb=True)
                else:
                    anneal_optim = lrs.split("_")[1]
                    run_info = get_run_info(results, size, optim, anneal=True, perturb=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token")

                if run_info == None:
                    print(f"Skipping {lrs} for OLMo {size}M")
                    continue
                ax = axs[col]
                for i, perturbation in enumerate(perturbations):
                    x = None
                    if match == "token":
                        x = [r["multiplier"] for r in run_info["perturbed"][perturbation].values()]
                    elif match == "loss":
                        x = [r["dclm_val"] for r in run_info["perturbed"][perturbation].values()]    
                    y = [r["dclm_perturbed"] for r in run_info["perturbed"][perturbation].values()]
                    ax.plot(x, y, label=f"std = {perturbation}", marker='o', color=cmap(i))

                ax.set_title(f"{LRS_MAP[lrs]}")
                ax.grid()

                if match == "token":
                    ax.set_xlabel("Tokens / Param", fontsize=FONTSIZE["AXIS"])
                    ax.set_xscale('log')
                else:
                    ax.set_xlabel("Pretrain Loss", fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel("Pretrain Loss after Perturbation", fontsize=FONTSIZE["AXIS"])

                ax.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
                
            # Use only one legend for the entire figure; use the handles/labels from the first axis
            handles, labels = axs[0].get_legend_handles_labels()
            axs[len(LRS)-1].legend(handles, labels, fontsize=FONTSIZE["LEGEND"])

            # if match == "token":
            #     plt.suptitle(f"OLMo-{size}M DCLM Val Loss after Perturbation vs Pretrain Token / Param (LRS)", fontsize=FONTSIZE["SUP_TITLE"])
            # elif match == "loss":
            #     plt.suptitle(f"OLMo-{size}M DCLM Val Loss after Perturbation vs DCLM Pretrain Val Loss (LRS)", fontsize=FONTSIZE["SUP_TITLE"])
                
            plt.tight_layout()  # type: ignore
            
            os.makedirs(os.path.join(RESULTS_DIR, f"plots/perturbation/lrs/"), exist_ok=True)
            plt.savefig(f"results/plots/perturbation/lrs/{size}m_{match}.png", bbox_inches="tight")
            plt.close()


# def make_lrs_plot(results):


#     n_sizes = len(SIZE)
#     fig, axs = plt.subplots(1, n_sizes, figsize=(FIG_WIDTH * n_sizes, 4), sharex=True)
#     optim = "adamw"

#     for col, size in enumerate(SIZE):
#         token_budget = TOKEN_LIST[size][-1]
#         ax = axs[col]
#         for pt_lrs in ["cosine", "wsd"]:
#             if pt_lrs == "cosine":
#                 run_info = get_run_info(results, size, optim, anneal=False, perturb=True)

#                 if run_info is None:
#                     continue

#                 # cosine_dclm_val = run_info["perturbed"][PERTURBATIONS_THRESHOLD][token_budget]["dclm_val"]
#                 # ax.axhline(
#                 #     cosine_dclm_val,
#                 #     label="Cosine",
#                 #     color="black",
#                 #     linestyle="-"
#                 # )
#                 cosine_dclm_val_pert = run_info["perturbed"][PERTURBATIONS_THRESHOLD][token_budget]["dclm_perturbed"]
#                 ax.axhline(
#                     cosine_dclm_val_pert,
#                     label="Cosine",
#                     color="black",
#                     linestyle="-"
#                 )


#             elif pt_lrs == "wsd":

#                 anneal_percent = ANNEAL_CONFIG["percent"]
#                 for anneal_optim in ANNEAL_CONFIG["anneal_optim"]:
#                     # anneal_dclm_val = np.empty_like(anneal_percent, dtype=float)
#                     anneal_dclm_pert = np.empty_like(anneal_percent, dtype=float)
#                     for j, p in enumerate(anneal_percent):
#                         # print(p, size)
#                         run_info = get_run_info(results, size, optim, anneal=True, anneal_percent=p, anneal_optim=anneal_optim, perturb=True, pt_lr=PT_LR[size][token_budget])

#                         if run_info is None:
#                             # anneal_dclm_val[j] = np.nan # type: ignore
#                             anneal_dclm_pert[j] = np.nan
#                             continue
                        
#                         # anneal_dclm_val[j] = run_info["perturbed"][PERTURBATIONS_THRESHOLD][token_budget]["dclm_val"]
#                         anneal_dclm_pert[j] = run_info["perturbed"][PERTURBATIONS_THRESHOLD][token_budget]["dclm_perturbed"]

#                     # ax.plot(
#                     #     anneal_percent,
#                     #     anneal_dclm_val,
#                     #     color=COLOR_MAP[anneal_optim], 
#                     #     marker="o",
#                     #     label=f"WSD ({OPTIM_MAP[anneal_optim]} Anneal)"
#                     # )
#                     ax.plot(
#                         anneal_percent,
#                         anneal_dclm_pert,
#                         color=COLOR_MAP[anneal_optim], 
#                         marker="o",
#                         linestyle="-",
#                         label=f"WSD ({OPTIM_MAP[anneal_optim]} Anneal)"
#                     )

#         ax.set_xlabel("Percentage of Total Steps for Annealing", fontsize=FONTSIZE["AXIS"])
#         if col == 0:
#             ax.set_ylabel(f"Pretrain DCLM Val Loss", fontsize=FONTSIZE["AXIS"])
#         ax.set_title(f"OLMo-{size}M", fontsize=FONTSIZE["TITLE"])

#         if col == len(SIZE)-1:
#             ax.legend(fontsize=FONTSIZE["LEGEND"])

#         ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
#         ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
#         # ax.set_xscale('log')
#         ax.grid(True)

#     fig.subplots_adjust(top=0.83)  # Make room for the suptitle before tight_layout
#     fig.suptitle(f"Pretrain DCLM Val Loss across Learning Rate Schedulers after Max Perturbation (0.025)", fontsize=FONTSIZE["SUP_TITLE"])
#     os.makedirs(os.path.join(RESULTS_DIR, "plots/perturbation"), exist_ok=True)
#     # plt.tight_layout(rect=[0, 0, 1, 0.93])
#     plt.savefig(os.path.join(RESULTS_DIR, f"plots/perturbation/lrs_all_size_token.png"), bbox_inches='tight')
#     plt.close()


def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    # perturbation_sam_vs_adamw_cosine(results)
    if args.plot == "lrs":
        perturbation_lrs(results)
    elif args.plot == "optim_cosine":
        perturbation_sam_vs_adamw_cosine(results)
    else:
        perturbation_sam_vs_adamw_cosine(results)
        perturbation_lrs(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perturbation Plotting Tool")
    parser.add_argument(
        "--plot",
        type=str,
        choices=["all", "optim_cosine", "lrs"],
        default="all",
        help="Which plot to generate: all, optim_cosine, lrs",
    )
    args = parser.parse_args()
    main(args)