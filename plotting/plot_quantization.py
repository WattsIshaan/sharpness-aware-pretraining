import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

from utils.plotting_globals import FIG_WIDTH, FONTSIZE, OPTIM_MAP, COLOR_MAP, LRS_MAP
from utils.config_globals import RESULTS_DIR, SIZE, OPTIM, BITS, TOKEN_LIST, PT_LR, ALL_LRS
from utils.helper import get_run_info

# def plot_quantization_plots_optim_token(results):

#     for size in SIZE:
#         fig, axs = plt.subplots(1, len(OPTIM), figsize=(FIG_WIDTH * len(OPTIM), 4), sharex="row", sharey="row")
        
#         lines = []
#         labels = []
#         for col, optim in enumerate(OPTIM):
#             ax = axs[col]
#             run_info = get_run_info(results, size, optim, model_type="hf", quantized=True)

#             if run_info is None:
#                 continue

#             # Baseline plot
#             x = [r["multiplier"] for r in run_info["pretrain"].values()]
#             y = [r["dclm_val"] for r in run_info["pretrain"].values()]
#             line, = ax.plot(
#                 x, y,
#                 marker="o",
#                 label="Baseline",
#                 linestyle='--',
#                 color="black"
#             )
#             if col == 0:
#                 lines.append(line)
#                 labels.append("Baseline")

#             # INT4 and INT8 quantization
#             for bit in BITS:
#                 q_run_info = run_info["quantized"].get(bit)
#                 if q_run_info is None:
#                     continue
#                 x_q = [r["multiplier"] for r in q_run_info.values()]
#                 y_q = [r["dclm_quant"] for r in q_run_info.values()]
#                 line, = ax.plot(
#                     x_q, y_q,
#                     marker="o",
#                     label=f"INT{bit}",
#                     color=COLOR_MAP[bit]
#                 )
#                 if col == 0:
#                     lines.append(line)
#                     labels.append(f"INT{bit}")

#             ax.set_title(f"{OPTIM_MAP[optim]} (Cosine)", fontsize=FONTSIZE["TITLE"])
#             ax.grid(True)
#             if col == 0:
#                 ax.set_ylabel(f"DCLM Val PPLX", fontsize=FONTSIZE["AXIS"])
#             if col == 1:
#                 ax.legend(lines, labels, fontsize=FONTSIZE["LEGEND"])
#             # ax.set_yscale('linear')
#             ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
#             ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
#             ax.set_xlabel("Tokens / Param", fontsize=FONTSIZE["AXIS"])
#             ax.set_xscale('log')

#         # Move legend to the right outside the plot
#         fig.suptitle(f"OLMo-{size}M | DCLM Val Loss vs Tokens / Params (Quantized)", fontsize=FONTSIZE["SUP_TITLE"], y=1.02)
#         # plt.tight_layout(rect=[0, 0, 1, 1])  # Shrink plot so legend doesn't overlap
#         os.makedirs(os.path.join(RESULTS_DIR, "plots/quantization"), exist_ok=True)
#         plt.savefig(os.path.join(RESULTS_DIR, f"plots/quantization/{size}m_optim_token.png"), bbox_inches='tight')
#         plt.close()


def plot_quantization_plots_lrs_token(results, anneal_match="token"):

    for size in SIZE:
        fig, axs = plt.subplots(1, len(BITS), figsize=(FIG_WIDTH * len(BITS), 4), sharex="row", sharey="row")
        optim = "adamw"
        
        for col, bit in enumerate(BITS):
            ax = axs[col]
            for lrs in ALL_LRS:
                if "cosine" in lrs:
                    base_optim = lrs.split("_")[1]
                    if base_optim == "sam":
                        run_info = get_run_info(results, size, base_optim, anneal=False, model_type="hf", quantized=True)
                    else:
                        run_info = get_run_info(results, size, base_optim, anneal=False, model_type="hf", quantized=True)
                else:
                    anneal_optim = lrs.split("_")[1]
                    if optim == "sam":
                        run_info = get_run_info(results, size, optim, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match=anneal_match, model_type="hf", quantized=True)
                    else:
                        run_info = get_run_info(results, size, optim, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token", model_type="hf", quantized=True)

                if run_info is None:
                    continue

                # Baseline plot
                if lrs == "cosine_adamw":
                    x = [r["multiplier"] for r in run_info["pretrain"].values()]
                    y = [r["dclm_val"] for r in run_info["pretrain"].values()]
                    ax.plot(
                        x, y,
                        marker="o",
                        label="Baseline",
                        linestyle='--',
                        color="black"
                    )

                # INT4 and INT8 quantization
                q_run_info = run_info["quantized"].get(bit)
                if q_run_info is None:
                    continue
                x_q = [r["multiplier"] for r in q_run_info.values()]
                y_q = [r["dclm_quant"] for r in q_run_info.values()]
                line, = ax.plot(
                    x_q, y_q,
                    marker="o",
                    label=LRS_MAP[lrs],
                    color=COLOR_MAP[lrs]
                )

            ax.set_title(f"INT{bit}", fontsize=FONTSIZE["TITLE"])
            ax.grid(True)
            if col == 0:
                ax.set_ylabel(f"Pretrain Loss", fontsize=FONTSIZE["AXIS"])
            if col == len(BITS)-1:
                ax.legend(fontsize=FONTSIZE["LEGEND"])
            # ax.set_yscale('linear')
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            ax.set_xlabel("Tokens / Param", fontsize=FONTSIZE["AXIS"])
            ax.set_xscale('log')

        # Move legend to the right outside the plot
        # fig.suptitle(f"OLMo-{size}M | DCLM Val Loss vs Tokens / Params (Quantized) across Learning Rate Schedulers", fontsize=FONTSIZE["SUP_TITLE"], y=1.02)
        # plt.tight_layout(rect=[0, 0, 1, 1])  # Shrink plot so legend doesn't overlap
        os.makedirs(os.path.join(RESULTS_DIR, "plots/quantization"), exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"plots/quantization/{size}m_lrs_{anneal_match}.png"), bbox_inches='tight')
        plt.close()



def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    if args.plot == "optim":
        if args.type == "token":
            plot_quantization_plots_optim_token(results)
        elif args.type == "compute":
            raise ValueError("Compute type not supported for optimization plots")
    elif args.plot == "lrs":
        if args.type == "token":
            plot_quantization_plots_lrs_token(results, anneal_match="token")
        elif args.type == "compute":
            raise ValueError("Compute type not supported for learning rate scheduler plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="optim", choices=["optim", "lrs"])
    parser.add_argument("--type", type=str, default="token", choices=["token", "compute"])

    args = parser.parse_args()
    main(args)