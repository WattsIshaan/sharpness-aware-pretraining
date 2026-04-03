import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import json
from scipy.spatial import ConvexHull
import numpy as np
import argparse

from utils.config_globals import SIZE, CPT_DATASET, OPTIM, TOKEN_LIST, RESULTS_DIR, LRS, TOKEN_PER_PARAM_MAP, ALL_LRS
from utils.plotting_globals import OPTIM_MAP, FONTSIZE, FIG_WIDTH, FIG_HEIGHT, COLOR_MAP, LRS_MAP, CPT_DATASET_MAP, MARKERS, XLABEL, YLABEL, LEGEND_PARAM, ALPHA
from utils.helper import get_run_info

YLIM = {
    150: {
        "starcoder": (0, 2.5),
        "musicpile" : (0, 1.55),
        "tulu": (0, 2.6),
        "gsm8k": (0, 10),
        "siqa": (0, 1.6),
        "stackmathqa": (0, 10),
    },
    60: {
        "starcoder": (0, 2.8),
        "musicpile" : (0, 1.7),
        "tulu": (0, 2.8),
        "gsm8k": (0, 1.5),
        "siqa": (0, 1.6),
        "stackmathqa": (0, 10),
    },
    20: {
        "starcoder": (0, 3.1),
        "musicpile" : (0, 2),
        "tulu": (0, 3.5),
        "gsm8k": (0, 2),
        "siqa": (0, 10),
        "stackmathqa": (0, 1.8),
    }
}
XLIM = {
    150: {
        "starcoder": (0, 4.25),
        "musicpile" : (0, 4),
        "tulu": (0, 3.5),
        "gsm8k": (0, 3.7),
        "siqa": (0, 5),
        "stackmathqa": (0, 4.5),
    },
    60: {
        "starcoder": (0, 4.5),
        "musicpile" : (0, 4.5),
        "tulu": (0, 3.8),
        "gsm8k": (0, 3.8),
        "siqa": (0, 4.2),
        "stackmathqa": (0, 4.5),
    },
    20: {
        "starcoder": (0, 5.5),
        "musicpile" : (0, 5.5),
        "tulu": (0, 4.2),
        "gsm8k": (0, 5),
        "siqa": (0, 5),
        "stackmathqa": (0, 5.5),
    }
}


def pareto_optim_size(results):
    print("Plotting optim size")

    for token_per_param in TOKEN_PER_PARAM_MAP:
        if len(TOKEN_PER_PARAM_MAP[token_per_param]) != len(SIZE):
            continue

        # print(f"Plotting for token_per_param = {token_per_param}")

        for cpt_dataset in CPT_DATASET:
            fig, axs = plt.subplots(1, len(SIZE), figsize=(FIG_WIDTH, FIG_HEIGHT))
            # fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, 4))

            tmp = dict()
            for col, size in enumerate(SIZE):
                tmp[size] = dict()
                ax = axs[col]
                for idx, optim in enumerate(OPTIM):
                    run_info = get_run_info(results, size, optim, cpt_dataset)
                    if run_info is None:
                        print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]} for {cpt_dataset.capitalize()}")
                        continue

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    dclm_val = []
                    cpt_val = []
                    used_cpt_lrs = []

                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][TOKEN_PER_PARAM_MAP[token_per_param][size]]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][TOKEN_PER_PARAM_MAP[token_per_param][size]][cpt_dataset]

                                    if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:
                                        if size == 20 and cpt_dataset == "starcoder":
                                            if x_val > 5.1:
                                                continue
                                        dclm_val.append(x_val)
                                        cpt_val.append(y_val)
                                        used_cpt_lrs.append(cpt_lr)
                                except:
                                    continue

                    ax.scatter(dclm_val, cpt_val, color=COLOR_MAP[optim], alpha=ALPHA, marker=MARKERS[idx])

                    if len(dclm_val) > 2:
                        points = np.empty((len(dclm_val), 2))
                        points[:, 0] = dclm_val
                        points[:, 1] = cpt_val
                        hull = ConvexHull(points)
                        # print(hull)
                        h = np.append(hull.vertices, hull.vertices[0])
                        # print(hull.vertices, hull.vertices[0])
                        # h = hull.vertices
                        new_points = np.empty((len(h), 2))
                        new_points[:, 0] = points[h, 0]
                        new_points[:, 1] = points[h, 1]
                        # Rotate so that first row has the smallest first column value
                        min_idx = np.argmin(new_points[:, 0])
                        new_points = np.roll(new_points, -min_idx, axis=0)

                        for i, num in enumerate(range(len(new_points)-1)):
                            if new_points[i][0] > new_points[i+1][0]:
                                new_points = new_points[:i+1, :]
                                break
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[optim], label=OPTIM_MAP[optim], marker=MARKERS[idx], alpha=ALPHA)

                
                ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)
                
                tmp[size]["dclm_val"] = (min(dclm_val), max(dclm_val)) # type: ignore
                tmp[size]["cpt_val"] = (min(cpt_val), max(cpt_val)) # type: ignore
                # if size == 20 and cpt_dataset == "starcoder":
                #     # Fix ValueError: setting xticks with float step
                #     x_min, x_max = ax.get_xlim()
                #     ax.set_xticks(np.arange(round(x_min, 1), round(x_max, 1) + 0.2, 0.2))
                # if size == 60 and cpt_dataset == "starcoder":
                #     y_min, y_max = ax.get_ylim()
                #     ax.set_yticks(np.arange(round(y_min, 2), round(y_max, 2) + 0.05, 0.05))

            y_span = float('-inf')
            x_span = float('-inf')
            for size in SIZE:
                x_span = max(x_span, tmp[size]["dclm_val"][1] - tmp[size]["dclm_val"][0])
                y_span = max(y_span, tmp[size]["cpt_val"][1] - tmp[size]["cpt_val"][0])
            

            for col, size in enumerate(SIZE):
                ax = axs[col]
                x_mid = (tmp[size]["dclm_val"][0] + tmp[size]["dclm_val"][1]) / 2
                y_mid = (tmp[size]["cpt_val"][0] + tmp[size]["cpt_val"][1]) / 2
                ax.set_xlim((x_mid - x_span / 2)*0.99, (x_mid + x_span / 2)*1.03)
                ax.set_ylim((y_mid - y_span / 2)*0.99, (y_mid + y_span / 2)*1.01)

            handles, labels = axs[len(SIZE)-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
            plt.tight_layout()

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/size/{token_per_param}"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/size/{token_per_param}/{cpt_dataset}_optim_token.pdf"), bbox_inches='tight')
            plt.close()


def pareto_lrs_size(results):

    print("Plotting lrs size")
    for token_per_param in TOKEN_PER_PARAM_MAP:
        if len(TOKEN_PER_PARAM_MAP[token_per_param]) != len(SIZE):
            continue

        # print(f"Plotting for token_per_param = {token_per_param}")
        optim = "adamw"

        for cpt_dataset in CPT_DATASET:
            fig, axs = plt.subplots(1, len(SIZE), figsize=(FIG_WIDTH, FIG_HEIGHT))
            # fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, 4))

            tmp = dict()
            for col, size in enumerate(SIZE):
                tmp[size] = dict()
                ax = axs[col]
                for idx, lrs in enumerate(LRS):

                    if lrs == "cosine":
                        run_info = get_run_info(results, size, optim, cpt_dataset, anneal=False)
                    else:
                        anneal_optim = lrs.split("_")[1]
                        run_info = get_run_info(results, size, optim, cpt_dataset, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token")
                    if run_info is None:
                        print(f"Skipping OLMo {size}M wwith {LRS_MAP[lrs]} for {cpt_dataset.capitalize()}")
                        continue

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    dclm_val = []
                    cpt_val = []
                    used_cpt_lrs = []

                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][TOKEN_PER_PARAM_MAP[token_per_param][size]]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][TOKEN_PER_PARAM_MAP[token_per_param][size]][cpt_dataset]

                                    if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:
                                        if size == 20 and cpt_dataset == "starcoder":
                                            if x_val > 5.1:
                                                continue
                                        dclm_val.append(x_val)
                                        cpt_val.append(y_val)
                                        used_cpt_lrs.append(cpt_lr)
                                except:
                                    continue

                    ax.scatter(dclm_val, cpt_val, color=COLOR_MAP[lrs], alpha=ALPHA, marker=MARKERS[idx])

                    if len(dclm_val) > 2:
                        points = np.empty((len(dclm_val), 2))
                        points[:, 0] = dclm_val
                        points[:, 1] = cpt_val
                        hull = ConvexHull(points)
                        # print(hull)
                        h = np.append(hull.vertices, hull.vertices[0])
                        # print(hull.vertices, hull.vertices[0])
                        # h = hull.vertices
                        new_points = np.empty((len(h), 2))
                        new_points[:, 0] = points[h, 0]
                        new_points[:, 1] = points[h, 1]
                        # Rotate so that first row has the smallest first column value
                        min_idx = np.argmin(new_points[:, 0])
                        new_points = np.roll(new_points, -min_idx, axis=0)

                        for i, num in enumerate(range(len(new_points)-1)):
                            if new_points[i][0] > new_points[i+1][0]:
                                new_points = new_points[:i+1, :]
                                break
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[lrs], label=LRS_MAP[lrs], marker=MARKERS[idx], alpha=ALPHA)

                
                ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)
                
                tmp[size]["dclm_val"] = (min(dclm_val), max(dclm_val)) # type: ignore
                tmp[size]["cpt_val"] = (min(cpt_val), max(cpt_val)) # type: ignore
                # if size == 20 and cpt_dataset == "starcoder":
                #     # Fix ValueError: setting xticks with float step
                #     x_min, x_max = ax.get_xlim()
                #     ax.set_xticks(np.arange(round(x_min, 1), round(x_max, 1) + 0.2, 0.2))
                # if size == 60 and cpt_dataset == "starcoder":
                #     y_min, y_max = ax.get_ylim()
                #     ax.set_yticks(np.arange(round(y_min, 2), round(y_max, 2) + 0.05, 0.05))

            y_span = float('-inf')
            x_span = float('-inf')
            for size in SIZE:
                x_span = max(x_span, tmp[size]["dclm_val"][1] - tmp[size]["dclm_val"][0])
                y_span = max(y_span, tmp[size]["cpt_val"][1] - tmp[size]["cpt_val"][0])
            

            for col, size in enumerate(SIZE):
                ax = axs[col]
                x_mid = (tmp[size]["dclm_val"][0] + tmp[size]["dclm_val"][1]) / 2
                y_mid = (tmp[size]["cpt_val"][0] + tmp[size]["cpt_val"][1]) / 2
                ax.set_xlim((x_mid - x_span / 2)*0.99, (x_mid + x_span / 2)*1.03)
                ax.set_ylim((y_mid - y_span / 2)*0.99, (y_mid + y_span / 2)*1.01)

            handles, labels = axs[len(SIZE)-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
            plt.tight_layout()

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/size/{token_per_param}"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/size/{token_per_param}/{cpt_dataset}_lrs_token.pdf"), bbox_inches='tight')
            plt.close()



def pareto_optim_dataset(results):

    print("Plotting optim dataset")
    thresholds = dict()
    for size in SIZE:
        for token_budget in TOKEN_LIST[size]:
            fig, axs = plt.subplots(1, len(CPT_DATASET), figsize=(FIG_WIDTH, FIG_HEIGHT))
            for col, cpt_dataset in enumerate(CPT_DATASET):
                ax = axs[col]
                for idx, optim in enumerate(OPTIM):
                    run_info = get_run_info(results, size, optim, cpt_dataset)
                    if run_info is None:
                        print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]} for {cpt_dataset.capitalize()}")
                        continue

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    dclm_val = []
                    cpt_val = []
                    used_cpt_lrs = []

                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget][cpt_dataset]

                                    if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:

                                        dclm_val.append(x_val)
                                        cpt_val.append(y_val)
                                        used_cpt_lrs.append(cpt_lr)
                                except:
                                    continue

                    ax.scatter(dclm_val, cpt_val, marker=MARKERS[idx], color=COLOR_MAP[optim], alpha=ALPHA)

                    if len(dclm_val) > 2:
                        points = np.empty((len(dclm_val), 2))
                        points[:, 0] = dclm_val
                        points[:, 1] = cpt_val
                        hull = ConvexHull(points)
                        # print(hull)
                        h = np.append(hull.vertices, hull.vertices[0])
                        # print(hull.vertices, hull.vertices[0])
                        # h = hull.vertices
                        new_points = np.empty((len(h), 2))
                        new_points[:, 0] = points[h, 0]
                        new_points[:, 1] = points[h, 1]
                        # Rotate so that first row has the smallest first column value
                        min_idx = np.argmin(new_points[:, 0])
                        new_points = np.roll(new_points, -min_idx, axis=0)

                        for i, num in enumerate(range(len(new_points)-1)):
                            if new_points[i][0] > new_points[i+1][0]:
                                new_points = new_points[:i+1, :]
                                break
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[optim], label = OPTIM_MAP[optim], marker=MARKERS[idx], alpha=ALPHA)

                ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
                ax.set_title(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["TITLE"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            # fig.suptitle(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["SUP_TITLE"])
            plt.tight_layout() # type: ignore
            # Place legend outside the figure on the right
            handles, labels = axs[len(CPT_DATASET)-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/dataset/"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/dataset/{size}m_optim_token_{int(token_budget * 1000 / size)}.pdf"), bbox_inches='tight')
            plt.close()     


def pareto_lrs_dataset(results):

    print("Plotting lrs dataset")
    optim = "adamw"
    for size in SIZE:
        for token_budget in TOKEN_LIST[size]:
            fig, axs = plt.subplots(1, len(CPT_DATASET), figsize=(FIG_WIDTH, FIG_HEIGHT))
            for col, cpt_dataset in enumerate(CPT_DATASET):
                ax = axs[col]
                for idx, lrs in enumerate(LRS):
                    if lrs == "cosine":
                        run_info = get_run_info(results, size, optim, cpt_dataset, anneal=False)
                    else:
                        anneal_optim = lrs.split("_")[1]
                        run_info = get_run_info(results, size, optim, cpt_dataset, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token")
                    if run_info is None:
                        print(f"Skipping OLMo {size}M wwith {LRS_MAP[lrs]} for {cpt_dataset.capitalize()}")
                        continue

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    dclm_val = []
                    cpt_val = []
                    used_cpt_lrs = []

                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget][cpt_dataset]

                                    if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:

                                        dclm_val.append(x_val)
                                        cpt_val.append(y_val)
                                        used_cpt_lrs.append(cpt_lr)
                                except:
                                    continue

                    ax.scatter(dclm_val, cpt_val, marker=MARKERS[idx], color=COLOR_MAP[lrs], alpha=ALPHA)

                    if len(dclm_val) > 2:
                        points = np.empty((len(dclm_val), 2))
                        points[:, 0] = dclm_val
                        points[:, 1] = cpt_val
                        hull = ConvexHull(points)
                        # print(hull)
                        h = np.append(hull.vertices, hull.vertices[0])
                        # print(hull.vertices, hull.vertices[0])
                        # h = hull.vertices
                        new_points = np.empty((len(h), 2))
                        new_points[:, 0] = points[h, 0]
                        new_points[:, 1] = points[h, 1]
                        # Rotate so that first row has the smallest first column value
                        min_idx = np.argmin(new_points[:, 0])
                        new_points = np.roll(new_points, -min_idx, axis=0)

                        for i, num in enumerate(range(len(new_points)-1)):
                            if new_points[i][0] > new_points[i+1][0]:
                                new_points = new_points[:i+1, :]
                                break
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[lrs], label = LRS_MAP[lrs], marker=MARKERS[idx], alpha=ALPHA)

                ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
                ax.set_title(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["TITLE"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            # fig.suptitle(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["SUP_TITLE"])
            plt.tight_layout() # type: ignore
            # Place legend outside the figure on the right
            handles, labels = axs[len(CPT_DATASET)-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/dataset/"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/dataset/{size}m_lrs_token_{int(token_budget * 1000 / size)}.pdf"), bbox_inches='tight')
            plt.close()   


# def make_tradeoff_pareto_all(results, anneal_match="token", single=False):

#     thresholds = dict()
#     for size in SIZE:
#         if single:
#             tokens_list = [TOKEN_LIST[size][-1]]
#         else:
#             tokens_list = TOKEN_LIST[size]
#         thresholds[size] = dict()
#         for cpt_dataset in CPT_DATASET:
#             print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
#             if single:
#                 fig, axs = plt.subplots(1, 1, figsize=(FIG_WIDTH // 2, 2.5))
#                 axs = [axs]
#             else:
#                 fig, axs = plt.subplots(1, len(tokens_list), figsize=(2 * FIG_WIDTH, 2.5), sharey=True, sharex=True)
#             global_cpt_min = []
#             for col, t in enumerate(tokens_list):
#                 ax = axs[col]

#                 for config in ALL_LRS:
#                     cpt_min = float('inf')
#                     optim = config.split("_")[1]
#                     if "cosine" in config:
#                         run_info = get_run_info(results, size, optim, cpt_dataset, anneal=False)
#                     elif "wsd" in config:
#                         if optim == "sam":
#                             run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=optim, anneal_match="token", cpt_dataset=cpt_dataset)
#                         else:
#                             run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=optim, anneal_match="token", cpt_dataset=cpt_dataset)
#                     else:
#                         raise ValueError(f"Invalid config: {config}")

#                     if run_info is None:
#                         continue

#                     cpt_lrs = sorted(run_info["cpt"].keys())
#                     dclm_val = []
#                     cpt_val = []
#                     used_cpt_lrs = []

#                     for cpt_lr in cpt_lrs:
#                         cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
#                         for cpt_wd in cpt_wds:
#                             cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
#                             for cpt_bs in cpt_bss:
#                                 try:
#                                     x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t]["dclm_val"]
#                                     y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t][cpt_dataset]

#                                     if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:
#                                         dclm_val.append(x_val)
#                                         cpt_val.append(y_val)
#                                         used_cpt_lrs.append(cpt_lr)
#                                         cpt_min = min(y_val, cpt_min)
#                                 except:
#                                     continue

#                     ax.scatter(dclm_val, cpt_val, label = LRS_MAP[config], marker="o", color=COLOR_MAP[config])
#                     global_cpt_min.append(cpt_min)
#                     # if lrs == "wsd_adamw" or lrs == "wsd_sam" and size == 150:
#                     #     for x, y, lr in zip(dclm_val, cpt_val, used_cpt_lrs):
#                     #         ax.text(x, y, f"{lr}", fontsize=8, ha='right', va='bottom', color='black')

#                     if len(dclm_val) > 2:
#                         points = np.empty((len(dclm_val), 2))
#                         points[:, 0] = dclm_val
#                         points[:, 1] = cpt_val
#                         hull = ConvexHull(points)
#                         # print(hull)
#                         h = np.append(hull.vertices, hull.vertices[0])
#                         # print(hull.vertices, hull.vertices[0])
#                         # h = hull.vertices
#                         new_points = np.empty((len(h), 2))
#                         new_points[:, 0] = points[h, 0]
#                         new_points[:, 1] = points[h, 1]
#                         # Rotate so that first row has the smallest first column value
#                         min_idx = np.argmin(new_points[:, 0])
#                         new_points = np.roll(new_points, -min_idx, axis=0)

#                         for i, num in enumerate(range(len(new_points)-1)):
#                             if new_points[i][0] > new_points[i+1][0]:
#                                 new_points = new_points[:i+1, :]
#                                 break
                                
#                         ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[config])

#                 ax.set_xlabel("Pretrain Loss", fontsize=FONTSIZE["AXIS"])
#                 if col == 0:
#                     ax.set_ylabel(f"Fine-tuning Loss", fontsize=FONTSIZE["AXIS"])
                
#                 if not single:
#                     ax.set_title(f"Tokens / Param = {int(t * 1000 / size)}", fontsize=FONTSIZE["TITLE"])

#                 ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
#                 ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
#                 ax.grid(True)

#             cpt_threshold = max(global_cpt_min)
#             thresholds[size][cpt_dataset] = cpt_threshold
#             for col, t in enumerate(tokens_list):
#                 ax = axs[col]
#                 # cpt_threshold = cpt_min * (1 + TRADEOFF_THRESHOLD[size][cpt_dataset])
                
#                 if not single:
#                     ax.axhline(cpt_threshold, linestyle='--', color='black', label="FT Threshold")

#             # fig.suptitle(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["SUP_TITLE"])
#             plt.tight_layout() # type: ignore

#             os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
#             handles, labels = axs[len(tokens_list)-1].get_legend_handles_labels()
#             if not single:
#                 fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
#             plt.tight_layout() # type: ignore
#             if single:
#                 plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto//{size}m/{cpt_dataset}_all_single_{anneal_match}.pdf"), bbox_inches='tight')
#             else:
#                 plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_all_{anneal_match}.pdf"), bbox_inches='tight')
#             plt.close()
    
#     if not single:
#         with open(os.path.join(RESULTS_DIR, "thresholds_all_token.json"), "w") as file:
#             json.dump(thresholds, file, indent=4)
   

def pareto_optim_token(results):
    print("Plotting optim token")

    thresholds = dict()
    for size in SIZE:
        tokens_list = TOKEN_LIST[size]
        thresholds[size] = dict()
        for cpt_dataset in CPT_DATASET:
            print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            fig, axs = plt.subplots(1, len(tokens_list), figsize=(2 * FIG_WIDTH, FIG_HEIGHT), sharey=True, sharex=True)
            # Ensure axs is iterable
            if len(tokens_list) == 1:
                axs = [axs]

            
            global_cpt_min = []
            for col, t in enumerate(tokens_list):
                ax = axs[col]

                for idx, optim in enumerate(OPTIM):
                    cpt_min = float('inf')
                    run_info = get_run_info(results, size, optim, cpt_dataset)
                    if run_info is None:
                        print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]} for {cpt_dataset.capitalize()}")
                        continue

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    dclm_val = []
                    cpt_val = []
                    used_cpt_lrs = []

                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t][cpt_dataset]

                                    if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:

                                        dclm_val.append(x_val)
                                        cpt_val.append(y_val)
                                        used_cpt_lrs.append(cpt_lr)
                                        cpt_min = min(y_val, cpt_min)
                                except:
                                    continue

                    ax.scatter(dclm_val, cpt_val, marker=MARKERS[idx], color=COLOR_MAP[optim], alpha=ALPHA)
                    global_cpt_min.append(cpt_min)
                    # ax.scatter(tmp["wd0"]["x"], tmp["wd0"]["y"], label = OPTIM_MAP[optim] + " wd0" , marker="*")
                    # ax.scatter(tmp["bs32"]["x"], tmp["bs32"]["y"], label = OPTIM_MAP[optim] + " wd1e-1", marker="v")
                    # ax.scatter(tmp["bs128"]["x"], tmp["bs128"]["y"], label = OPTIM_MAP[optim] + " bs128", marker="p")
                    # if optim == "adamw" and size == 60:
                    #     for x, y, lr in zip(dclm_val, cpt_val, used_cpt_lrs):
                    #         ax.text(x, y, f"{lr}", fontsize=8, ha='right', va='bottom', color='black')

                    # assert len(dclm_val) > 0, f"No points found for {cpt_dataset.capitalize()} for OLMo {size}M with {OPTIM_MAP[optim]}"

                    if len(dclm_val) > 2:
                        points = np.empty((len(dclm_val), 2))
                        points[:, 0] = dclm_val
                        points[:, 1] = cpt_val
                        hull = ConvexHull(points)
                        # print(hull)
                        h = np.append(hull.vertices, hull.vertices[0])
                        # print(hull.vertices, hull.vertices[0])
                        # h = hull.vertices
                        new_points = np.empty((len(h), 2))
                        new_points[:, 0] = points[h, 0]
                        new_points[:, 1] = points[h, 1]
                        # Rotate so that first row has the smallest first column value
                        min_idx = np.argmin(new_points[:, 0])
                        new_points = np.roll(new_points, -min_idx, axis=0)

                        for i, num in enumerate(range(len(new_points)-1)):
                            if new_points[i][0] > new_points[i+1][0]:
                                new_points = new_points[:i+1, :]
                                break
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[optim], label = OPTIM_MAP[optim], marker=MARKERS[idx], alpha=ALPHA)

                ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"Tokens={t}B", fontsize=FONTSIZE["TITLE"])

                # ax.legend(fontsize=FONTSIZE["LEGEND"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            cpt_threshold = max(global_cpt_min)
            thresholds[size][cpt_dataset] = cpt_threshold
            # for col, t in enumerate(tokens_list):
            #     ax = axs[col]
                # ax.axhline(cpt_threshold, linestyle='--', color='black', label="FT Threshold")

            handles, labels = axs[len(tokens_list)-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
            plt.tight_layout() # type: ignore

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_optim_token.pdf"), bbox_inches='tight')
            plt.close()
        
    with open(os.path.join(RESULTS_DIR, "thresholds_optim_token.json"), "w") as file:
        json.dump(thresholds, file, indent=4)


def pareto_optim_cpt_token(results):
    print("Plotting optim cpt token")

    CPT_TOKEN_LIST = [2, 5, 10, 20]
    for size in [60]:
        token_budget = TOKEN_LIST[size][-1]
        for cpt_dataset in ["starcoder"]:
            print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            fig, axs = plt.subplots(1, len(CPT_TOKEN_LIST), figsize=(2 * FIG_WIDTH, 3), sharey=True, sharex=True)
            # Ensure axs is iterable

            for col, cpt_token in enumerate(CPT_TOKEN_LIST):
                ax = axs[col]
                for idx, optim in enumerate(OPTIM):
                    run_info = get_run_info(results, size, optim, cpt_dataset, cpt_tokens=cpt_token)
                    if run_info is None:
                        print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]} for {cpt_dataset.capitalize()}")
                        continue

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    dclm_val = []
                    cpt_val = []
                    used_cpt_lrs = []

                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget][cpt_dataset]

                                    # if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:
                                    if x_val <= 5 and y_val <= 3.5:
                                        dclm_val.append(x_val)
                                        cpt_val.append(y_val)
                                        used_cpt_lrs.append(cpt_lr)
                                except:
                                    continue

                    ax.scatter(dclm_val, cpt_val, marker=MARKERS[idx], color=COLOR_MAP[optim], alpha=ALPHA)

                    if len(dclm_val) > 2:
                        points = np.empty((len(dclm_val), 2))
                        points[:, 0] = dclm_val
                        points[:, 1] = cpt_val
                        hull = ConvexHull(points)
                        # print(hull)
                        h = np.append(hull.vertices, hull.vertices[0])
                        # print(hull.vertices, hull.vertices[0])
                        # h = hull.vertices
                        new_points = np.empty((len(h), 2))
                        new_points[:, 0] = points[h, 0]
                        new_points[:, 1] = points[h, 1]
                        # Rotate so that first row has the smallest first column value
                        min_idx = np.argmin(new_points[:, 0])
                        new_points = np.roll(new_points, -min_idx, axis=0)

                        for i, num in enumerate(range(len(new_points)-1)):
                            if new_points[i][0] > new_points[i+1][0]:
                                new_points = new_points[:i+1, :]
                                break
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[optim], label = OPTIM_MAP[optim], marker=MARKERS[idx], alpha=ALPHA)

                ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"FT Tokens={cpt_token}M", fontsize=FONTSIZE["TITLE"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            handles, labels = axs[len(CPT_TOKEN_LIST)-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
            plt.tight_layout() # type: ignore
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_optim_cpt_token.pdf"), bbox_inches='tight')
            plt.close()

def pareto_optim_compute(results):

    print("Plotting optim compute")

    thresholds = dict()
    for size in SIZE:
        tokens_list = TOKEN_LIST[size]
        thresholds[size] = dict()
        for cpt_dataset in CPT_DATASET:
            fig, axs = plt.subplots(1, len(tokens_list)-1, figsize=(2 * FIG_WIDTH, FIG_HEIGHT), sharey=True, sharex=True)
            # Ensure axs is iterable
            if len(tokens_list) == 1:
                axs = [axs]

            global_cpt_min = []
            for col in range(len(tokens_list)-1):

                ax = axs[col]
                for idx, optim in enumerate(OPTIM):
                    cpt_min = float('inf')
                    run_info = get_run_info(results, size, optim, cpt_dataset)
                    if run_info is None:
                        print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]} for {cpt_dataset.capitalize()}")
                        continue

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    dclm_val = []
                    cpt_val = []
                    used_cpt_lrs = []

                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val, y_val = None, None
                                    if optim == "sam":
                                        x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][tokens_list[col]]["dclm_val"]
                                        y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][tokens_list[col]][cpt_dataset]
                                    elif optim == "adamw":
                                        x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][tokens_list[col+1]]["dclm_val"]
                                        y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][tokens_list[col+1]][cpt_dataset]
                                    
                                    assert x_val is not None and y_val is not None

                                    if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:

                                        dclm_val.append(x_val)
                                        cpt_val.append(y_val)
                                        used_cpt_lrs.append(cpt_lr)
                                        cpt_min = min(y_val, cpt_min)
                                except:
                                    continue

                    ax.scatter(dclm_val, cpt_val, marker=MARKERS[idx], color=COLOR_MAP[optim], alpha=ALPHA)
                    global_cpt_min.append(min(cpt_val))
                    # ax.scatter(tmp["wd0"]["x"], tmp["wd0"]["y"], label = OPTIM_MAP[optim] + " wd0" , marker="*")
                    # ax.scatter(tmp["bs32"]["x"], tmp["bs32"]["y"], label = OPTIM_MAP[optim] + " bs32", marker="v")
                    # ax.scatter(tmp["bs128"]["x"], tmp["bs128"]["y"], label = OPTIM_MAP[optim] + " bs128", marker="p")

                    points = np.empty((len(dclm_val), 2))
                    points[:, 0] = dclm_val
                    points[:, 1] = cpt_val
                    hull = ConvexHull(points)
                    # print(hull)
                    h = np.append(hull.vertices, hull.vertices[0])
                    # print(hull.vertices, hull.vertices[0])
                    # h = hull.vertices
                    new_points = np.empty((len(h), 2))
                    new_points[:, 0] = points[h, 0]
                    new_points[:, 1] = points[h, 1]
                    # Rotate so that first row has the smallest first column value
                    min_idx = np.argmin(new_points[:, 0])
                    new_points = np.roll(new_points, -min_idx, axis=0)

                    for i, num in enumerate(range(len(new_points)-1)):
                        if new_points[i][0] > new_points[i+1][0]:
                            new_points = new_points[:i+1, :]
                            break
                            
                    ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[optim], label = OPTIM_MAP[optim], marker=MARKERS[idx], alpha=ALPHA)

                ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"FLOPs={int(tokens_list[col] / min(tokens_list))}X", fontsize=FONTSIZE["TITLE"])

                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            cpt_threshold = max(global_cpt_min)
            thresholds[size][cpt_dataset] = cpt_threshold
            # for col in range(len(tokens_list)-1):
            #     ax = axs[col]
            #     ax.axhline(cpt_threshold, linestyle='--', color='black', label="FT Threshold")

            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
            plt.tight_layout()

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_optim_compute.pdf"), bbox_inches='tight')
            plt.close()

    with open(os.path.join(RESULTS_DIR, "thresholds_optim_compute.json"), "w") as file:
        json.dump(thresholds, file, indent=4)





def pareto_lrs(results, anneal_match="token", single=False):
    print("Plotting lrs")

    thresholds = dict()
    optim = "adamw"
    for size in SIZE:
        if single:
            tokens_list = [TOKEN_LIST[size][-1]]
        else:
            tokens_list = TOKEN_LIST[size]
        thresholds[size] = dict()
        for cpt_dataset in CPT_DATASET:
            print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            if single:
                fig, axs = plt.subplots(1, 1, figsize=(FIG_WIDTH // 2, FIG_HEIGHT))
                axs = [axs]
            else:
                fig, axs = plt.subplots(1, len(tokens_list), figsize=(2 * FIG_WIDTH, FIG_HEIGHT), sharey=True, sharex=True)
            global_cpt_min = []
            for col, t in enumerate(tokens_list):
                ax = axs[col]

                for idx, lrs in enumerate(LRS):
                    cpt_min = float('inf')
                    if lrs == "cosine":
                        run_info = get_run_info(results, size, optim, cpt_dataset, anneal=False)
                    else:
                        anneal_optim = lrs.split("_")[1]
                        if optim == "sam":
                            run_info = get_run_info(results, size, optim, cpt_dataset, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match=anneal_match)
                        else:
                            run_info = get_run_info(results, size, optim, cpt_dataset, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token")

                    if run_info is None:
                        continue

                    cpt_lrs = sorted(run_info["cpt"].keys())
                    dclm_val = []
                    cpt_val = []
                    used_cpt_lrs = []

                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t][cpt_dataset]

                                    if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:
                                        dclm_val.append(x_val)
                                        cpt_val.append(y_val)
                                        used_cpt_lrs.append(cpt_lr)
                                        cpt_min = min(y_val, cpt_min)
                                except:
                                    continue

                    ax.scatter(dclm_val, cpt_val, marker=MARKERS[idx], color=COLOR_MAP[lrs], alpha=ALPHA)
                    global_cpt_min.append(cpt_min)
                    # if lrs == "wsd_adamw" or lrs == "wsd_sam" and size == 150:
                    #     for x, y, lr in zip(dclm_val, cpt_val, used_cpt_lrs):
                    #         ax.text(x, y, f"{lr}", fontsize=8, ha='right', va='bottom', color='black')

                    if len(dclm_val) > 2:
                        points = np.empty((len(dclm_val), 2))
                        points[:, 0] = dclm_val
                        points[:, 1] = cpt_val
                        hull = ConvexHull(points)
                        # print(hull)
                        h = np.append(hull.vertices, hull.vertices[0])
                        # print(hull.vertices, hull.vertices[0])
                        # h = hull.vertices
                        new_points = np.empty((len(h), 2))
                        new_points[:, 0] = points[h, 0]
                        new_points[:, 1] = points[h, 1]
                        # Rotate so that first row has the smallest first column value
                        min_idx = np.argmin(new_points[:, 0])
                        new_points = np.roll(new_points, -min_idx, axis=0)

                        for i, num in enumerate(range(len(new_points)-1)):
                            if new_points[i][0] > new_points[i+1][0]:
                                new_points = new_points[:i+1, :]
                                break
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[lrs], label = LRS_MAP[lrs], marker=MARKERS[idx], alpha=ALPHA)

                ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
                
                if not single:
                    ax.set_title(f"Tokens={t}B", fontsize=FONTSIZE["TITLE"])

                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            cpt_threshold = max(global_cpt_min)
            thresholds[size][cpt_dataset] = cpt_threshold
            # for col, t in enumerate(tokens_list):
            #     ax = axs[col]
            #     # cpt_threshold = cpt_min * (1 + TRADEOFF_THRESHOLD[size][cpt_dataset])
                
            #     if not single:
            #         ax.axhline(cpt_threshold, linestyle='--', color='black', label="FT Threshold")

            # fig.suptitle(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["SUP_TITLE"])
            plt.tight_layout() # type: ignore

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
            handles, labels = axs[-1].get_legend_handles_labels()
            if not single:
                fig.legend(handles, labels, loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles), fontsize=FONTSIZE["LEGEND"])
            plt.tight_layout()
            if single:
                plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto//{size}m/{cpt_dataset}_lrs_single_{anneal_match}.pdf"), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_lrs_{anneal_match}.pdf"), bbox_inches='tight')
            plt.close()
    
    if not single:
        with open(os.path.join(RESULTS_DIR, f"thresholds_lrs_{anneal_match}.json"), "w") as file:
            json.dump(thresholds, file, indent=4)



def pareto_pt_lr(results):

    print("Plotting pt lr")

    thresholds = dict()
    optim = "adamw"
    for size in [60]:
        token_budget = TOKEN_LIST[size][-1]
        thresholds[size] = dict()
        for cpt_dataset in ["starcoder"]:
            print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True, sharex=True)
            # Ensure axs is iterable

            for idx, pt_lr in enumerate([1e-4, 3e-4, 6e-4]):
                run_info = get_run_info(results, size, optim, cpt_dataset, pt_lr=pt_lr)
                
                if run_info is None:
                    continue

                pt_loss = run_info["pretrain"][token_budget]["dclm_val"]

                cpt_lrs = sorted(run_info["cpt"].keys())
                dclm_val = []
                cpt_val = []
                used_cpt_lrs = []

                for cpt_lr in cpt_lrs:
                    cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                    for cpt_wd in cpt_wds:
                        cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                        for cpt_bs in cpt_bss:
                            try:
                                x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget]["dclm_val"]
                                y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget][cpt_dataset]

                                # if x_val <= XLIM[size][cpt_dataset][1] and y_val <= YLIM[size][cpt_dataset][1]:
                                if x_val <= 7 and y_val <= 2.8:
                                    dclm_val.append(x_val)
                                    cpt_val.append(y_val)
                                    used_cpt_lrs.append(cpt_lr)
                            except:
                                continue

                ax.scatter(
                    dclm_val,
                    cpt_val,
                    marker=MARKERS[idx],
                    color=COLOR_MAP[pt_lr],
                    alpha=ALPHA
                )

                if len(dclm_val) > 2:
                    points = np.empty((len(dclm_val), 2))
                    points[:, 0] = dclm_val
                    points[:, 1] = cpt_val
                    hull = ConvexHull(points)
                    # print(hull)
                    h = np.append(hull.vertices, hull.vertices[0])
                    # print(hull.vertices, hull.vertices[0])
                    # h = hull.vertices
                    new_points = np.empty((len(h), 2))
                    new_points[:, 0] = points[h, 0]
                    new_points[:, 1] = points[h, 1]
                    # Rotate so that first row has the smallest first column value
                    min_idx = np.argmin(new_points[:, 0])
                    new_points = np.roll(new_points, -min_idx, axis=0)

                    for i, num in enumerate(range(len(new_points)-1)):
                        if new_points[i][0] > new_points[i+1][0]:
                            new_points = new_points[:i+1, :]
                            break
                    
                    if pt_lr == 3e-4:
                        label = f"{pt_lr:.0e}*"
                    else:
                        label = f"{pt_lr:.0e}"
                    ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[pt_lr], label=label, marker=MARKERS[idx], alpha=ALPHA)

            ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax.legend(
                title="Pretrain LR",
                fontsize=FONTSIZE["LEGEND"],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title_fontsize=FONTSIZE["LEGEND"]  # Set title fontsize
            )
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            ax.grid(True)

            plt.tight_layout() # type: ignore

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/pt_lr"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/pt_lr/{size}m_{cpt_dataset}.pdf"), bbox_inches='tight')
            plt.close()
        


def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    if args.plot == "optim":
        if args.type == "token":
            pareto_optim_token(results)
        elif args.type == "compute":
            pareto_optim_compute(results)
    elif args.plot == "lrs":
        if args.type == "compute":
            raise NotImplementedError("Compute plots not implemented yet")
        pareto_lrs(results, anneal_match=args.type, single=args.single)
    elif args.plot == "dataset":
        pareto_optim_dataset(results)
        pareto_lrs_dataset(results)
    elif args.plot == "pt_lr":
        pareto_pt_lr(results)
    elif args.plot == "cpt_token":
        pareto_optim_cpt_token(results)
    elif args.plot == "size":
        pareto_optim_size(results)
        pareto_lrs_size(results)
    elif args.plot == "all":
        # pareto_optim_dataset(results)
        # pareto_optim_size(results)
        # pareto_lrs_size(results)
        pareto_optim_token(results)
        # pareto_optim_compute(results)
        # pareto_lrs(results, anneal_match="token", single=True)
        # pareto_lrs(results, anneal_match="token", single=False)
        # pareto_pt_lr(results)
        # pareto_lrs_dataset(results)
        # pareto_optim_cpt_token(results)
    else:
        raise ValueError(f"Invalid plot: {args.plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="all", choices=["optim", "lrs", "pt_lr", "dataset", "size", "cpt_token", "all"])
    parser.add_argument("--type", type=str, default="token", choices=["token", "compute"])
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    main(args)