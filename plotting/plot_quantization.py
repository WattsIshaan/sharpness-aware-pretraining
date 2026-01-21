import json
import os
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting_globals import FIG_WIDTH, FONTSIZE, OPTIM_MAP, COLOR_MAP
from utils.config_globals import RESULTS_DIR, SIZE, OPTIM, BITS, TOKEN_LIST, PT_LR
from utils.helper import get_run_info

def plot_quantization_plots_token(results):

    for size in SIZE:
        fig, axs = plt.subplots(1, len(OPTIM), figsize=(FIG_WIDTH * len(OPTIM), 4), sharex="row", sharey="row")
        
        lines = []
        labels = []
        for col, optim in enumerate(OPTIM):
            ax = axs[col]
            run_info = get_run_info(results, size, optim, model_type="hf", quantized=True)

            if run_info is None:
                continue

            # Baseline plot
            x = [r["multiplier"] for r in run_info["pretrain"].values()]
            y = [r["dclm_val"] for r in run_info["pretrain"].values()]
            line, = ax.plot(
                x, y,
                marker="o",
                label="Baseline",
                linestyle='--',
                color="black"
            )
            if col == 0:
                lines.append(line)
                labels.append("Baseline")

            # INT4 and INT8 quantization
            for bit in BITS:
                q_run_info = run_info["quantized"].get(bit)
                if q_run_info is None:
                    continue
                x_q = [r["multiplier"] for r in q_run_info.values()]
                y_q = [r["dclm_quant"] for r in q_run_info.values()]
                line, = ax.plot(
                    x_q, y_q,
                    marker="o",
                    label=f"INT{bit}",
                    color=COLOR_MAP[bit]
                )
                if col == 0:
                    lines.append(line)
                    labels.append(f"INT{bit}")

            if optim == "adamw":
                token_budget = TOKEN_LIST[size][-1]

                for anneal_optim in OPTIM:
                    wsd_run_info = get_run_info(results, size, optim, model_type="hf", anneal=True, pt_lr=PT_LR[size][token_budget], quantized=True, anneal_optim=anneal_optim, anneal_percent=10)
                    if wsd_run_info is not None:
                        x = [int(1000*token_budget/size) for _ in BITS[:-1]]
                        y = [wsd_run_info["quantized"][b][token_budget]["dclm_quant"] for b in BITS[:-1]]
                        line = ax.scatter(x, y, color=COLOR_MAP[4])
                        for i, (x_point, y_point, bit) in enumerate(zip(x, y, BITS[:-1])):
                            label_text = f"WSD ({OPTIM_MAP[anneal_optim]})"
                            ax.text(
                                x_point, y_point,
                                label_text,
                                fontsize=FONTSIZE["LEGEND"]-4,
                                ha='left',
                                va='bottom',
                                color="black",
                                # rotation=0,
                                # bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5)
                            )
                        # labels.append(f"WSD ({OPTIM_MAP[anneal_optim]} Anneal)")

            ax.set_title(f"{OPTIM_MAP[optim]}", fontsize=FONTSIZE["TITLE"])
            ax.grid(True)
            if col == 0:
                ax.set_ylabel(f"DCLM Val PPLX", fontsize=FONTSIZE["AXIS"])
            if col == 1:
                ax.legend(lines, labels, fontsize=FONTSIZE["LEGEND"])
            # ax.set_yscale('linear')
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            ax.set_xlabel("Tokens / Param", fontsize=FONTSIZE["AXIS"])
            ax.set_xscale('log')

        # Move legend to the right outside the plot
        fig.suptitle(f"OLMo-{size}M | DCLM Val Loss vs Tokens / Params (Quantized)", fontsize=FONTSIZE["SUP_TITLE"], y=1.02)
        # plt.tight_layout(rect=[0, 0, 1, 1])  # Shrink plot so legend doesn't overlap
        os.makedirs(os.path.join(RESULTS_DIR, "plots/quantization"), exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"plots/quantization/optim_{size}m_token.png"), bbox_inches='tight')
        plt.close()


def main():

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    plot_quantization_plots_token(results)

if __name__ == "__main__":
    main()