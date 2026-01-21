import enum
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting_globals import FIG_WIDTH, FONTSIZE
from utils.config_globals import RESULTS_DIR, SIZE
from utils.helper import get_run_info


TUNING_LR = {
    150: {
        15: [3e-3, 1e-3, 6e-4],
        30: [1e-3, 6e-4],
        60: [1e-3, 6e-4, 3e-4],
        120: [6e-4, 3e-4]
    },
    60: {
        12: [3e-3, 1e-3, 6e-4],
        24: [6e-4, 1e-3],
        48: [1e-3, 3e-4, 6e-4],
        96: [6e-4, 3e-4],
        192: [6e-4, 3e-4, 1e-4]
    },
    20: {
        4: [1e-2, 3e-3, 1e-3],
        8: [3e-3, 1e-3],
        16: [3e-3, 1e-3],
        32: [3e-3, 1e-3, 6e-4],
        64: [1e-3, 6e-4]
    }
}


def plot_pt_lr_tuning(results):

    optim = "adamw"
    fig, axs = plt.subplots(1, len(SIZE), figsize=(3 * FIG_WIDTH, 4))
    for col, size in enumerate(SIZE):
        ax = axs[col]

        for t in TUNING_LR[size]:
            x = []
            y = []
            lrs = sorted(TUNING_LR[size][t])
            for lr in lrs:
                run_info = get_run_info(results, size, optim, pt_lr=lr)

                if run_info is None:
                    continue

                x.append(lr)
                y.append(run_info["pretrain"][t]["dclm_val"])
            
            ax.plot(x, y, marker="o", label=f"{int(t*1000/size)}")
            min_idx = np.argmin(y)
            min_lr = lrs[min_idx]
            min_loss = y[min_idx]
            ax.plot(min_lr, min_loss, marker="*", color="k", ms=14, label=None, zorder=10)

        ax.set_title(f"OLMo-{size}M", fontsize=FONTSIZE["TITLE"])
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate", fontsize=FONTSIZE["AXIS"])
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
        if col == 0:
            ax.set_ylabel("Pretrain DCLM Val Loss", fontsize=FONTSIZE["AXIS"])
        ax.legend(title="D/N", loc="best", fontsize=FONTSIZE["LEGEND"])
        ax.grid(True)

    plt.tight_layout()
    os.makedirs("results/plots/additional", exist_ok=True)
    plt.suptitle("Pretrain Learning Rate Tuning", fontsize=FONTSIZE["SUP_TITLE"])
    plt.tight_layout()
    plt.savefig("results/plots/additional/pt_lr_tuning.png")
    plt.close()


def main():

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    plot_pt_lr_tuning(results)



if __name__ == "__main__":
    main()