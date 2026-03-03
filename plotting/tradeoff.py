from token import OP
import matplotlib.pyplot as plt
import os
import json
from scipy.spatial import ConvexHull
import numpy as np
import argparse
from utils.config_globals import SIZE, CPT_DATASET, OPTIM, TOKEN_LIST, RESULTS_DIR, LRS
from utils.plotting_globals import OPTIM_MAP, FONTSIZE, FIG_WIDTH, FIG_HEIGHT, COLOR_MAP, CPT_DATASET_MAP, MARKERS, XLABEL, ALPHA, LEGEND_PARAM, YLABEL, LRS_MAP
from utils.helper import get_run_info

def tradeoff_summary_optim(results):

    with open(os.path.join(RESULTS_DIR, "thresholds_optim_token.json"), "r") as file:
        thresholds = json.load(file)

    for size in SIZE:
        fig, axs = plt.subplots(1, len(CPT_DATASET), figsize=(FIG_WIDTH, 2.75), sharex=True)
        for col, cpt_dataset in enumerate(CPT_DATASET):
            ax = axs[col]

            # print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            tokens_list = TOKEN_LIST[size]
            plotting_data = dict()
            for optim in OPTIM:
                plotting_data[optim] = {
                    "x": [],
                    "y": []
                }

            for i, t in enumerate(tokens_list):
                for optim in OPTIM:
                    run_info = get_run_info(results, size, optim, cpt_dataset)
                    if run_info is None:
                        print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]} for {cpt_dataset.capitalize()}")
                        continue

                    plotting_data[optim]["x"].append(run_info["pretrain"][t]["dclm_val"])
                    plotting_data[optim]["tmp_x"] = []
                    plotting_data[optim]["tmp_y"] = []

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t][cpt_dataset]
                                    plotting_data[optim]["tmp_x"].append(x_val)
                                    plotting_data[optim]["tmp_y"].append(y_val)
                                except:
                                    continue
                
                
                # print(f"Minimum CPT Val --> {cpt_min}, Threshold --> {cpt_threshold}")

                for idx, optim in enumerate(OPTIM):
                    points = np.empty((len(plotting_data[optim]["tmp_x"]), 2))
                    points[:, 0] = plotting_data[optim]["tmp_x"]
                    points[:, 1] = plotting_data[optim]["tmp_y"]
                    hull = ConvexHull(points)

                    h = np.append(hull.vertices, hull.vertices[0])

                    new_points = np.empty((len(h), 2))
                    new_points[:, 0] = points[h, 0]
                    new_points[:, 1] = points[h, 1]
                    # Rotate so that first row has the smallest first column value
                    min_idx = np.argmin(new_points[:, 0])
                    new_points = np.roll(new_points, -min_idx, axis=0)

                    # for i, _ in enumerate(range(len(new_points)-1)):
                    #     if new_points[i][0] > new_points[i+1][0]:
                    #         new_points = new_points[:i+1, :]
                    #         break

                    hull_x, hull_y = new_points[:, 0], new_points[:, 1]
                    x_at_threshold = None
                    cpt_threshold = thresholds[str(size)][cpt_dataset]
                    while x_at_threshold is None:
                        for idx in range(len(hull_x) - 1):
                            x1, x2 = hull_x[idx], hull_x[idx + 1]
                            y1, y2 = hull_y[idx], hull_y[idx + 1]
                            if (y1 - cpt_threshold) * (y2 - cpt_threshold) <= 0:
                                if y2 != y1:
                                    alpha = (cpt_threshold - y1) / (y2 - y1)
                                    x_at_threshold = x1 + alpha * (x2 - x1)
                                else:
                                    x_at_threshold = x1
                                break
                    
                    assert x_at_threshold is not None, f"No threshold found for {OPTIM_MAP[optim]}, {t}B, {size}M, {cpt_dataset}"
                    
                    plotting_data[optim]["y"].append(x_at_threshold)

            for idx, optim in enumerate(OPTIM):
                ax.plot(
                    plotting_data[optim]["x"],
                    plotting_data[optim]["y"],
                    marker=MARKERS[idx],
                    color=COLOR_MAP[optim],
                    label=OPTIM_MAP[optim],
                    alpha=ALPHA
                )
            ax.grid(True)
            ax.set_title(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["TITLE"])
            ax.set_xlabel(XLABEL["PT_LOSS_BASE"], fontsize=FONTSIZE["AXIS"])
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])

        plt.tight_layout()
        handles, labels = axs[len(CPT_DATASET)-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
        axs[0].set_ylabel(YLABEL["PT_LOSS_FT"], fontsize=FONTSIZE["AXIS"])
        os.makedirs(os.path.join(RESULTS_DIR, f"plots/tradeoff/"), exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"plots/tradeoff/{size}m_tradeoff_optim.pdf"), bbox_inches='tight')
        plt.close()


def tradeoff_summary_lrs(results):

    with open(os.path.join(RESULTS_DIR, "thresholds_lrs_token.json"), "r") as file:
        thresholds = json.load(file)

    for size in SIZE:
        fig, axs = plt.subplots(1, len(CPT_DATASET), figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
        for col, cpt_dataset in enumerate(CPT_DATASET):
            print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            ax = axs[col]

            # print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            tokens_list = TOKEN_LIST[size]
            plotting_data = dict()
            for lrs in LRS:
                plotting_data[lrs] = {
                    "x": [],
                    "y": []
                }

            for t in tokens_list:
                for idx, lrs in enumerate(LRS):
                    
                    if lrs == "cosine":
                        run_info = get_run_info(results, size, "adamw", cpt_dataset, anneal=False)
                    else:
                        anneal_optim = lrs.split("_")[1]
                        run_info = get_run_info(results, size, "adamw", cpt_dataset, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token")

                    if run_info is None:
                        print(f"Skipping OLMo {size}M wwith {LRS_MAP[lrs]} for {cpt_dataset.capitalize()}")
                        continue

                    plotting_data[lrs]["x"].append(run_info["pretrain"][t]["dclm_val"])
                    plotting_data[lrs]["tmp_x"] = []
                    plotting_data[lrs]["tmp_y"] = []

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t][cpt_dataset]
                                    plotting_data[lrs]["tmp_x"].append(x_val)
                                    plotting_data[lrs]["tmp_y"].append(y_val)
                                except:
                                    continue
                
                
                # print(f"Minimum CPT Val --> {cpt_min}, Threshold --> {cpt_threshold}")

                for idx, lrs in enumerate(LRS):
                    points = np.empty((len(plotting_data[lrs]["tmp_x"]), 2))
                    points[:, 0] = plotting_data[lrs]["tmp_x"]
                    points[:, 1] = plotting_data[lrs]["tmp_y"]
                    hull = ConvexHull(points)

                    h = np.append(hull.vertices, hull.vertices[0])

                    new_points = np.empty((len(h), 2))
                    new_points[:, 0] = points[h, 0]
                    new_points[:, 1] = points[h, 1]
                    # Rotate so that first row has the smallest first column value
                    min_idx = np.argmin(new_points[:, 0])
                    new_points = np.roll(new_points, -min_idx, axis=0)

                    # for i, _ in enumerate(range(len(new_points)-1)):
                    #     if new_points[i][0] > new_points[i+1][0]:
                    #         new_points = new_points[:i+1, :]
                    #         break

                    hull_x, hull_y = new_points[:, 0], new_points[:, 1]
                    x_at_threshold = None
                    cpt_threshold = thresholds[str(size)][cpt_dataset]
                    while x_at_threshold is None:
                        for idx in range(len(hull_x) - 1):
                            x1, x2 = hull_x[idx], hull_x[idx + 1]
                            y1, y2 = hull_y[idx], hull_y[idx + 1]
                            if (y1 - cpt_threshold) * (y2 - cpt_threshold) <= 0:
                                if y2 != y1:
                                    alpha = (cpt_threshold - y1) / (y2 - y1)
                                    x_at_threshold = x1 + alpha * (x2 - x1)
                                else:
                                    x_at_threshold = x1
                                break
                    
                    assert x_at_threshold is not None, f"No threshold found for {LRS_MAP[lrs]}, {t}B, {size}M, {cpt_dataset}"
                    
                    plotting_data[lrs]["y"].append(x_at_threshold)

            for idx, lrs in enumerate(LRS):
                ax.plot(
                    plotting_data[lrs]["x"],
                    plotting_data[lrs]["y"],
                    marker=MARKERS[idx],
                    color=COLOR_MAP[lrs],
                    label=LRS_MAP[lrs],
                    alpha=ALPHA
                )
            ax.grid(True)
            ax.set_title(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["TITLE"])
            ax.set_xlabel(XLABEL["PT_LOSS_BASE"], fontsize=FONTSIZE["AXIS"])
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])

        plt.tight_layout()
        handles, labels = axs[len(CPT_DATASET)-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
        axs[0].set_ylabel(YLABEL["PT_LOSS_FT"], fontsize=FONTSIZE["AXIS"])
        os.makedirs(os.path.join(RESULTS_DIR, f"plots/tradeoff/"), exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"plots/tradeoff/{size}m_tradeoff_lrs.pdf"), bbox_inches='tight')
        plt.close()






def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    if args.plot == "optim":
        tradeoff_summary_optim(results)
    elif args.plot == "lrs":
        tradeoff_summary_lrs(results)
    else:
        tradeoff_summary_optim(results)
        tradeoff_summary_lrs(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="all", choices=["optim", "lrs"])
    args = parser.parse_args()
    main(args)