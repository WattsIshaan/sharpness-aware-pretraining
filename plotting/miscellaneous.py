import matplotlib.pyplot as plt
import os
import json
import argparse

from utils.config_globals import SIZE, OPTIM, RESULTS_DIR, LRS, ANNEAL_CONFIG, TOKEN_LIST, PERTURBATIONS, PERTURBATION_THRESHOLD, ALL_LRS
from utils.plotting_globals import FONTSIZE, FIG_WIDTH, FIG_HEIGHT, COLOR_MAP, LRS_MAP, LEGEND_PARAM, ALPHA, GRID_ALPHA, MARKERS, OPTIM_MAP
from utils.helper import get_run_info
from scipy.spatial import ConvexHull
import numpy as np
from utils.plotting_globals import XLABEL, YLABEL
from matplotlib.ticker import NullLocator
from pareto import XLIM, YLIM


def perturbation_lrs_steps(results):

    for size in [60]:
        perturbation = 0.020
        
        

        optim = "adamw"
        token_budget = TOKEN_LIST[size][-1]
        for cpt_dataset in ["starcoder"]:

            fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
            ax1, ax2 = axs

            for col, lrs in enumerate(LRS):
                if lrs == "cosine":
                    run_info = get_run_info(results, size, optim, perturb=True)
                else:
                    anneal_optim = lrs.split("_")[1]
                    run_info = get_run_info(results, size, optim, anneal=True, perturb=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token")

                if run_info == None:
                    print(f"Skipping {lrs} for OLMo {size}M")
                    continue

                x = [r["multiplier"] for r in run_info["perturbed"][perturbation].values()]
                y = [r["dclm_perturbed"] for r in run_info["perturbed"][perturbation].values()]
                ax1.plot(x, y, label=LRS_MAP[lrs], marker=MARKERS[col], color=COLOR_MAP[lrs], alpha=ALPHA)

            ax1.grid()
            ax1.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
            ax1.set_xscale('log')
            ax1.set_ylabel(YLABEL["PT_LOSS_PERTURBED"], fontsize=FONTSIZE["AXIS"])
            ax1.minorticks_on()
            ax1.yaxis.set_minor_locator(NullLocator())
            ax1.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax1.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax1.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
            ax1.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
            ax1.set_title("Perturbation", fontsize=FONTSIZE["TITLE"])

            if size == 150:
                ax1.set_xlim(ax1.get_xlim()[0], 1010)



            for col, lrs in enumerate(LRS):
                if lrs == "cosine":
                    run_info = get_run_info(results, size, optim, cpt_dataset, anneal=False)
                else:
                    anneal_optim = lrs.split("_")[1]
                    run_info = get_run_info(results, size, optim, cpt_dataset, anneal=True, anneal_percent=10, anneal_optim=anneal_optim, anneal_match="token")
                
                if run_info is None:
                    print(f"Skipping {lrs} for OLMo {size}M")
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

                ax2.scatter(dclm_val, cpt_val, marker=MARKERS[col], color=COLOR_MAP[lrs], alpha=ALPHA)

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
                            
                    ax2.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[lrs], label = LRS_MAP[lrs], marker=MARKERS[col], alpha=ALPHA)

            ax2.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax2.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax2.set_title("FT-PT Frontier", fontsize=FONTSIZE["TITLE"])
            ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax2.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            ax2.grid(True)
            plt.tight_layout()
 
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles))
            plt.tight_layout()  # type: ignore

            plt.savefig(os.path.join(RESULTS_DIR, f"plots/miscellaneous/perturbation_pareto_lrs.pdf"), bbox_inches="tight")
            plt.close()


def pareto_lrs_steps(results):

    perturbation = 0.025
    optim = "adamw"
    for size in [60]:
        token_budget = TOKEN_LIST[size][-1]
        for cpt_dataset in ["starcoder"]:
            print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            fig, axs = plt.subplots(1, 3, figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
            
            for idx, anneal_optim in enumerate(ANNEAL_CONFIG["anneal_optim"]):
                x = []
                y = []
                for percent in ANNEAL_CONFIG["percent"]:
                    run_info = get_run_info(results, size, optim, anneal=True, perturb=True, anneal_percent=percent, anneal_optim=anneal_optim, anneal_match="token")
                    if run_info is None:
                        print(f"Skipping {percent}% with Anneal Optim {anneal_optim} for OLMo {size}M")
                        continue

                    x.append(percent)
                    y.append(run_info["perturbed"][perturbation][192]["dclm_perturbed"])

                axs[0].plot(x, y, label=LRS_MAP[f"wsd_{anneal_optim}"], marker=MARKERS[idx], color=COLOR_MAP[f"wsd_{anneal_optim}"], alpha=ALPHA)
            
            axs[0].set_title("Perturbation", fontsize=FONTSIZE["TITLE"])
            axs[0].grid()
            axs[0].set_xlabel(XLABEL["ANNEAL_PERCENT"], fontsize=FONTSIZE["AXIS"])
            axs[0].set_xscale('log')
            axs[0].minorticks_on()
            axs[0].yaxis.set_minor_locator(NullLocator())
            axs[0].grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
            axs[0].grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
            axs[0].xaxis.set_tick_params(which='minor', labelbottom=False)
            axs[0].tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
            axs[0].tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
            axs[0].legend(fontsize=FONTSIZE["LEGEND"])


            for idx, anneal_optim in enumerate(ANNEAL_CONFIG["anneal_optim"]):
                ax = axs[idx+1]
                for percent in ANNEAL_CONFIG["percent"]:
                    run_info = get_run_info(results, size, optim, cpt_dataset, anneal=True, anneal_percent=percent, anneal_optim=anneal_optim, anneal_match="token")
                    if run_info is None:
                        print(f"Skipping {percent}% with Anneal Optim {anneal_optim} for OLMo {size}M")
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

                    ax.scatter(dclm_val, cpt_val, marker=MARKERS[idx], color=COLOR_MAP[f"percent{percent}"], alpha=ALPHA)

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
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[f"percent{percent}"], label = f"{percent}%", marker=MARKERS[idx], alpha=ALPHA)

                ax.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                if idx == 0:
                    ax.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
                ax.set_title(LRS_MAP[f"wsd_{anneal_optim}"], fontsize=FONTSIZE["TITLE"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)
                ax.legend(fontsize=FONTSIZE["LEGEND"])
                plt.tight_layout()

            # handles, labels = axs[0].get_legend_handles_labels()
            # fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles))
            plt.tight_layout()  # type: ignore
            
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/miscellaneous/perturbation_pareto_steps.pdf"), bbox_inches="tight")
            plt.close()

        
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


def pt_lr_tuning(results):

    import matplotlib.pyplot as plt

    # Set colormap as viridis and create colors for the labels
    labels = [100, 200, 400, 800, 1600, 3200]
    cmap = plt.get_cmap('viridis', len(labels))

    optim = "adamw"
    fig, axs = plt.subplots(1, len(SIZE), figsize=(FIG_WIDTH * 2, 4))
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

            ratio = int(t*1000/size)
            ax.plot(x, y, marker="o", label=ratio, alpha=ALPHA, color=cmap(labels.index(ratio)))
            min_idx = np.argmin(y)
            min_lr = lrs[min_idx]
            min_loss = y[min_idx]
            ax.plot(min_lr, min_loss, marker="*", color="k", ms=14, label=None, zorder=10)

        ax.set_title(f"{size}M", fontsize=FONTSIZE["TITLE"])
        ax.set_xscale("log")
        ax.set_xlabel(XLABEL["LR"], fontsize=FONTSIZE["AXIS"])
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
        ax.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
        ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
        if col == 0:
            ax.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
        ax.grid(True)

    # Get all unique handles for the legend
    all_handles = []
    all_labels = []
    for ax in axs[::-1]:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in all_labels:
                all_handles.append(h)
                all_labels.append(l)
    fig.legend(all_handles, all_labels, title=XLABEL["TOKEN_RATIO"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(all_handles), fontsize=FONTSIZE["LEGEND"], title_fontsize=FONTSIZE["LEGEND"])
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"plots/miscellaneous/pt_lr_tuning.pdf"), bbox_inches="tight")
    plt.close()



def pareto_optim_pretrain(results):

    for size in [60]:
        fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax1, ax2 = axs

        for idx, optim in enumerate(OPTIM):
            run_info = get_run_info(results, size, optim)
            if run_info is None:
                print(f"Skipping OLMo {size}M wwith {OPTIM_MAP[optim]}")
                continue 

            x = [r["multiplier"] for r in run_info["pretrain"].values()]
            y = [r["dclm_val"] for r in run_info["pretrain"].values()]

            ax1.plot(
                x, y,
                label=OPTIM_MAP[optim],
                color=COLOR_MAP[optim],
                alpha=ALPHA,
                marker=MARKERS[idx]
            )

        ax1.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
        ax1.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
        ax1.set_xscale('log')
        ax1.minorticks_on()
        ax1.yaxis.set_minor_locator(NullLocator())
        ax1.set_title("Pretrain", fontsize=FONTSIZE["TITLE"])
        ax1.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
        ax1.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
        ax1.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
        ax1.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])

        token_budget = TOKEN_LIST[size][-1]
        cpt_dataset = "starcoder"
        for idx, optim in enumerate(OPTIM):
            run_info = get_run_info(results, size, optim, cpt_dataset)
            if run_info is None:
                print(f"Skipping {OPTIM_MAP[optim]} for OLMo {size}M")
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

            ax2.scatter(dclm_val, cpt_val, marker=MARKERS[idx], color=COLOR_MAP[optim], alpha=ALPHA)

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
                        
                ax2.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[optim], label = OPTIM_MAP[optim], marker=MARKERS[idx], alpha=ALPHA)

            ax2.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax2.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax2.set_title("FT-PT Frontier", fontsize=FONTSIZE["TITLE"])
            ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax2.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            ax2.grid(True)
            plt.tight_layout()
 
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=LEGEND_PARAM["BBOX_TO_ANCHOR"], ncol=len(handles))
        plt.tight_layout()  # type: ignore
        plt.savefig(os.path.join(RESULTS_DIR, f"plots/miscellaneous/pareto_optim_pretrain.png"), bbox_inches="tight")
        plt.close()


def pareto_ptlr_lrs_pert(results):

    thresholds = dict()
    optim = "adamw"
    
    for size in [60]:
        token_budget = TOKEN_LIST[size][-1]
        for cpt_dataset in ["starcoder"]:
            print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            fig, axs = plt.subplots(1, 3, figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
            ax1, ax2, ax3 = axs
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

                ax1.scatter(
                    dclm_val,
                    cpt_val,
                    marker=MARKERS[0],
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
                            
                    ax1.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[pt_lr], label=label, marker=MARKERS[0], alpha=ALPHA)

            ax1.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax1.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
            # ax1.set_title("FT-PT Frontier", fontsize=FONTSIZE["TITLE"])
            ax1.legend(
                title="Pretrain LR (Base Loss)",
                fontsize=FONTSIZE["LEGEND"],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title_fontsize=FONTSIZE["LEGEND"]  # Set title fontsize
            )
            ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax1.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            labels, handles = ax1.get_legend_handles_labels()
            ax1.legend(labels, handles, fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=(0.5, -0.7), ncol=len(handles), title_fontsize=FONTSIZE["LEGEND"])
            ax1.grid(True)



            perturbations = [p for p in PERTURBATIONS if p <= PERTURBATION_THRESHOLD]

            for col, lrs in enumerate(LRS):
                if lrs == "cosine":
                    run_info = get_run_info(results, size, 'adamw', perturb=True)
                    if run_info == None:
                        print(f"Skipping {lrs} for OLMo {size}M")
                        continue

                    x = []
                    y = []
                    for i, perturbation in enumerate(perturbations):
                        x.append(perturbation)
                        y.append(run_info["perturbed"][perturbation][token_budget]["dclm_perturbed"])
                    ax2.plot(x, y, label=LRS_MAP[lrs], marker=MARKERS[col], color=COLOR_MAP[lrs], alpha=ALPHA)
                        
                else:
                    anneal_optim = lrs.split("_")[1]
                    if anneal_optim == "sam":
                        continue
                    for percent in ANNEAL_CONFIG["percent"]:
                        run_info = get_run_info(results, size, 'adamw', anneal=True, perturb=True, anneal_percent=percent, anneal_optim=anneal_optim, anneal_match="token")
                        if run_info == None:
                            print(f"Skipping {percent}% with Anneal Optim {anneal_optim} for OLMo {size}M")
                            continue
                        x = []
                        y = []
                        for i, perturbation in enumerate(perturbations):
                            x.append(perturbation)
                            y.append(run_info["perturbed"][perturbation][token_budget]["dclm_perturbed"])
                        ax2.plot(x, y, label=LRS_MAP[lrs] + f" ({percent}%)", marker=MARKERS[col], color=COLOR_MAP[f"percent{percent}"], alpha=ALPHA)

                ax2.set_xlabel("Perturbation (Gamma)", fontsize=FONTSIZE["AXIS"])
                # ax2.set_xscale('log')
                ax2.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
                ax2.tick_params(axis='x', labelsize=FONTSIZE["TICKS"])
                ax2.tick_params(axis='y', labelsize=FONTSIZE["TICKS"])
                ax2.set_ylim(ax2.get_ylim()[0] - 0.025, 3.8)
                ax2.grid(True)

        
            for col, lrs in enumerate(LRS):
                if lrs == "cosine":
                    run_info = get_run_info(results, size, 'adamw', cpt_dataset, anneal=False)
                    if run_info is None:
                        print(f"Skipping {lrs} for OLMo {size}M")
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

                    ax3.scatter(dclm_val, cpt_val, marker=MARKERS[col], color=COLOR_MAP[lrs], alpha=ALPHA)

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
                                
                        ax3.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[lrs], label = LRS_MAP[lrs], marker=MARKERS[col], alpha=ALPHA)
                else:
                    anneal_optim = lrs.split("_")[1]
                    if anneal_optim == "sam":
                        continue
                    for percent in ANNEAL_CONFIG["percent"]:
                        
                        run_info = get_run_info(results, size, 'adamw', cpt_dataset, anneal=True, anneal_percent=percent, anneal_optim=anneal_optim, anneal_match="token")
                        if run_info is None:
                            print(f"Skipping {percent}% with Anneal Optim {anneal_optim} for OLMo {size}M")
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

                        ax3.scatter(dclm_val, cpt_val, marker=MARKERS[col], color=COLOR_MAP[f"percent{percent}"], alpha=ALPHA)

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
                                    
                            ax3.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[f"percent{percent}"], label = LRS_MAP[lrs] + f" ({percent}%)", marker=MARKERS[col], alpha=ALPHA)
                
            ax3.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax3.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
            # ax2.set_title("FT-PT Frontier", fontsize=FONTSIZE["TITLE"])
            ax3.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax3.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            ax3.grid(True)
            plt.tight_layout()
 
            handles, labels = ax2.get_legend_handles_labels()
            # ax2.legend(handles, labels, fontsize=FONTSIZE["LEGEND"], loc="lower center", bbox_to_anchor=(0.75, -0.3), ncol=len(handles))
            fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=(0.66, -0.15), ncol=len(handles))
            plt.tight_layout()  # type: ignore

            plt.savefig(os.path.join(RESULTS_DIR, f"plots/miscellaneous/pareto_ptlr_pert_lrs.pdf"), bbox_inches="tight")
            plt.close()


def pareto_pert_quant(results):
    
    for size in [60]:
        fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.25))
        ax1, ax2 = axs

        for idx, lrs in enumerate(ALL_LRS):
            if "cosine" in lrs:
                base_optim = lrs.split("_")[1]
                if base_optim == "sam":
                    run_info = get_run_info(results, size, base_optim, anneal=False, model_type="hf", quantized=True)
                else:
                    run_info = get_run_info(results, size, base_optim, anneal=False, model_type="hf", quantized=True)
            else:
                anneal_optim = lrs.split("_")[1]
                if anneal_optim == "sam":
                    run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=anneal_optim, model_type="hf", quantized=True)
                else:
                    run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=anneal_optim, model_type="hf", quantized=True)

            if run_info is None:
                continue

            # Baseline plot
            if lrs == "cosine_adamw":
                x = [r["multiplier"] for r in run_info["pretrain"].values()]
                y = [r["dclm_val"] for r in run_info["pretrain"].values()]
                ax1.plot(
                    x, y,
                    marker="x",
                    # label="Baseline",
                    linestyle='--',
                    color="black"
                )

            # INT4 and INT8 quantization
            q_run_info = run_info["quantized"].get(4)
            if q_run_info is None:
                continue
            x_q = [r["multiplier"] for r in q_run_info.values()]
            y_q = [r["dclm_quant"] for r in q_run_info.values()]
            line, = ax1.plot(
                x_q, y_q,
                marker=MARKERS[idx],
                label=LRS_MAP[lrs],
                color=COLOR_MAP[lrs],
                alpha=ALPHA
            )
        
            if size == 150:
                ax1.set_xlim(90, 1000)

            ax1.set_title("Quantized", fontsize=FONTSIZE["TITLE"])
            ax1.grid(True)
            ax1.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax1.minorticks_on()
            ax1.yaxis.set_minor_locator(NullLocator())
            ax1.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax1.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax1.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            ax1.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
            ax1.set_xscale('log')
        



        perturbation = PERTURBATION_THRESHOLD

        for col, lrs in enumerate(ALL_LRS):
            if "cosine" in lrs:
                base_optim = lrs.split("_")[1]
                if base_optim == "sam":
                    run_info = get_run_info(results, size, base_optim, anneal=False, perturb=True)
                else:
                    run_info = get_run_info(results, size, base_optim, anneal=False, perturb=True)
            else:
                anneal_optim = lrs.split("_")[1]
                if anneal_optim == "sam":
                    run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=anneal_optim, perturb=True)
                else:
                    run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=anneal_optim, perturb=True)

            if run_info is None:
                continue


            x = [r["multiplier"] for r in run_info["perturbed"][perturbation].values()]
            y = [r["dclm_perturbed"] for r in run_info["perturbed"][perturbation].values()]
            ax2.plot(x, y, label=LRS_MAP[lrs], marker=MARKERS[col], color=COLOR_MAP[lrs], alpha=ALPHA)
            ax2.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
            # ax2.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax2.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            ax2.set_xscale('log')
            ax2.grid(True)
            ax2.minorticks_on()
            ax2.yaxis.set_minor_locator(NullLocator())
            ax2.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax2.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax2.set_ylim(ax1.get_ylim()[0], 3.6)

            if size == 150:
                ax2.set_xlim(90, 1000)

            ax2.set_title("Perturbed", fontsize=FONTSIZE["TITLE"])

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=(0.5, -0.25), ncol=2)
        plt.tight_layout()  # type: ignore

        plt.savefig(os.path.join(RESULTS_DIR, f"plots/miscellaneous/pareto_pert_quant.pdf"), bbox_inches="tight")
        plt.close()


def pareto_optim_lrs_quant(results):

    for size in [60]:
        fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3))
        ax1, ax2 = axs

        cpt_dataset = "starcoder"
        token_budget = TOKEN_LIST[size][-1]

        for col, lrs in enumerate(ALL_LRS):
            if "cosine" in lrs:
                base_optim = lrs.split("_")[1]
                if base_optim == "sam":
                    run_info = get_run_info(results, size, base_optim, anneal=False, cpt_dataset=cpt_dataset)
                else:
                    run_info = get_run_info(results, size, base_optim, anneal=False, cpt_dataset=cpt_dataset)
            else:
                anneal_optim = lrs.split("_")[1]
                if anneal_optim == "sam":
                    run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=anneal_optim, cpt_dataset=cpt_dataset)
                else:
                    run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=anneal_optim, cpt_dataset=cpt_dataset)

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
                            x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget]["dclm_val"]
                            y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][token_budget][cpt_dataset]

                            if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:
                                dclm_val.append(x_val)
                                cpt_val.append(y_val)
                                used_cpt_lrs.append(cpt_lr)
                        except:
                            continue

            ax1.scatter(dclm_val, cpt_val, marker=MARKERS[col], color=COLOR_MAP[lrs], alpha=ALPHA)

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
                        
                ax1.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[lrs], label = LRS_MAP[lrs], marker=MARKERS[col], alpha=ALPHA)

        ax1.set_title("FT-PT Frontier", fontsize=FONTSIZE["TITLE"])
        ax1.grid(True)
        ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
        ax1.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
        ax1.set_xlabel(XLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
        ax1.set_ylabel(YLABEL["FT_LOSS"], fontsize=FONTSIZE["AXIS"])

        for col, lrs in enumerate(ALL_LRS):
            if "cosine" in lrs:
                base_optim = lrs.split("_")[1]
                if base_optim == "sam":
                    run_info = get_run_info(results, size, base_optim, anneal=False, quantized=True, model_type="hf")
                else:
                    run_info = get_run_info(results, size, base_optim, anneal=False, quantized=True, model_type="hf")
            else:
                anneal_optim = lrs.split("_")[1]
                if anneal_optim == "sam":
                    run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=anneal_optim, quantized=True, model_type="hf")
                else:
                    run_info = get_run_info(results, size, "adamw", anneal=True, anneal_percent=10, anneal_optim=anneal_optim, quantized=True, model_type="hf")

            if run_info is None:
                continue

            if lrs == "cosine_adamw":
                x = [r["multiplier"] for r in run_info["pretrain"].values()]
                y = [r["dclm_val"] for r in run_info["pretrain"].values()]
                ax2.plot(x, y, marker="x", linestyle='--', color="black", alpha=ALPHA)


            q_run_info = run_info["quantized"].get(4)
            if q_run_info is None:
                continue
            x_q = [r["multiplier"] for r in q_run_info.values()]
            y_q = [r["dclm_quant"] for r in q_run_info.values()]
            ax2.plot(x_q, y_q, marker=MARKERS[col], color=COLOR_MAP[lrs], alpha=ALPHA)

            ax2.set_xlabel(XLABEL["TOKEN_RATIO"], fontsize=FONTSIZE["AXIS"])
            ax2.set_ylabel(YLABEL["PT_LOSS"], fontsize=FONTSIZE["AXIS"])
            ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            ax2.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            ax2.set_xscale('log')
            ax2.grid(True)
            ax2.minorticks_on()
            ax2.yaxis.set_minor_locator(NullLocator())
            ax2.grid(which='minor', axis='both', linestyle='-', alpha=GRID_ALPHA)
            ax2.grid(which='major', axis='both', linestyle='-', alpha=GRID_ALPHA)
            

            if size == 150:
                ax2.set_xlim(90, 1000)

            ax2.set_title("Quantized", fontsize=FONTSIZE["TITLE"])

        handles, labels = ax1.get_legend_handles_labels()
        labels = [l.replace("SAWD", "SAWD*") for l in labels]
        # ax2.set_ylim(ax2.get_ylim()[0], 3.8)
        fig.legend(handles, labels, fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=(0.5, -0.25), ncol=2)
        plt.tight_layout()  # type: ignore

        plt.savefig(os.path.join(RESULTS_DIR, f"plots/miscellaneous/pareto_optim_lrs_quant.pdf"), bbox_inches="tight")
        plt.close()

def lr_schematic():

    T = 100.0
    steps = 1000*T
    x = np.linspace(0, T, int(steps))

    # Learning-rate settings
    lr_max = 3.0
    lr_min = 0.3
    lr_start = 0.0

    # Schedule parameters
    warmup_frac = 0.05
    decay_frac = 0.10

    warmup_end = warmup_frac * T
    decay_start = (1 - decay_frac) * T

    # --------------------
    # Shared warmup
    # --------------------
    def warmup_lr(t):
        return lr_start + (lr_max - lr_start) * (t / warmup_end)

    # --------------------
    # Cosine schedule
    # --------------------
    cosine_lr = np.zeros_like(x)

    for i, t in enumerate(x):
        if t < warmup_end:
            cosine_lr[i] = warmup_lr(t)
        else:
            progress = (t - warmup_end) / (T - warmup_end)
            cosine_lr[i] = lr_min + 0.5 * (lr_max - lr_min) * (
                1 + np.cos(np.pi * progress)
            )

    # --------------------
    # WSD schedule
    # --------------------
    wsd_lr = np.zeros_like(x)

    for i, t in enumerate(x):
        if t < warmup_end:
            wsd_lr[i] = warmup_lr(t)
        elif t < decay_start:
            wsd_lr[i] = lr_max
        else:
            progress = (t - decay_start) / (T - decay_start)
            wsd_lr[i] = lr_min + (lr_max - lr_min) * (1 - progress)

    # --------------------
    # Plot
    # --------------------
    plt.figure(figsize=(7, 4))

    plt.plot(x, cosine_lr, label=LRS_MAP["cosine"], color=COLOR_MAP["cosine"])
    plt.plot(x, wsd_lr, label=LRS_MAP["wsd_adamw"], color=COLOR_MAP["wsd_adamw"])

    # Highlight WSD decay phase
    # plt.axvspan(
    #     decay_start,
    #     T,
    #     alpha=0.15,
    #     # label="WSD decay phase"
    # )

    # # Vertical marker
    # plt.axvline(
    #     decay_start,
    #     linestyle="--",
    #     linewidth=1
    # )

    # # Annotation
    # plt.text(
    #     decay_start + 0.02,
    #     lr_max * 0.9,
    #     "WSD Decay",
    #     fontsize=10,
    #     verticalalignment="bottom"
    # )

    plt.xlabel("Number of Training Tokens (Billions)", fontsize=FONTSIZE["AXIS"])
    plt.ylabel("Learning Rate", fontsize=FONTSIZE["AXIS"])
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
    plt.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
    # plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=FONTSIZE["LEGEND"], loc=LEGEND_PARAM["LOC"], bbox_to_anchor=(0.5, -0.35), ncol=2, title_fontsize=FONTSIZE["LEGEND"])
    plt.savefig(os.path.join(RESULTS_DIR, f"plots/miscellaneous/lr_schematic.pdf"), bbox_inches="tight")
    plt.close()



def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)
    

    os.makedirs(os.path.join(RESULTS_DIR, f"plots/miscellaneous/"), exist_ok=True)

    if args.plot == "perturbation_lrs_steps":
        perturbation_lrs_steps(results)
    elif args.plot == "pareto_lrs_steps":
        pareto_lrs_steps(results)
    elif args.plot == "tune":
        pt_lr_tuning(results)
    elif args.plot == "optim_pretrain":
        pareto_optim_pretrain(results)
    elif args.plot == "ptlr_lrs_pert":
        # pareto_lrs_ptlr(results)
        pareto_ptlr_lrs_pert(results)
    elif args.plot == "pq":
        pareto_pert_quant(results)
    elif args.plot == "optim_lrs_quant":
        pareto_optim_lrs_quant(results)
    elif args.plot == "all":
        perturbation_lrs_steps(results)
        pareto_lrs_steps(results)
        pt_lr_tuning(results)
        pareto_optim_pretrain(results)
        pareto_ptlr_lrs_pert(results)
        pareto_pert_quant(results)
        pareto_optim_lrs_quant(results)
    elif args.plot == "lr_schematic":
        lr_schematic()
    else:
        raise ValueError(f"Invalid plot: {args.plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Miscellaneous Plotting Tool")
    parser.add_argument(
        "--plot",
        type=str,
        default="all",
        choices=["perturbation_lrs_steps", "pareto_lrs_steps", "tune", "optim_pretrain", "ptlr_lrs_pert", "lr_schematic", "all", "pq", "optim_lrs_quant"],
        help="Which plot to generate: perturbation_lrs_steps, pareto_lrs_steps, pt_lr_tuning, pareto_optim_pretrain, all",
    )
    args = parser.parse_args()
    main(args)