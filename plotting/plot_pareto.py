import matplotlib.pyplot as plt
import os
import json
from scipy.spatial import ConvexHull
import numpy as np
import argparse

from utils.config_globals import SIZE, CPT_DATASET, OPTIM, TOKEN_LIST, RESULTS_DIR, LRS
from utils.plotting_globals import OPTIM_MAP, FONTSIZE, FIG_WIDTH, COLOR_MAP, LRS_MAP, CPT_DATASET_MAP
from utils.helper import get_run_info

YLIM = {
    150: {
        "starcoder": (0, 2.5),
        "musicpile" : (0, 1.55),
        "tulu": (0, 3),
        "gsm8k": (0, 10),
        "siqa": (0, 1.6),
        "stackmathqa": (0, 10),
    },
    60: {
        "starcoder": (0, 2.8),
        "musicpile" : (0, 1.7),
        "tulu": (0, 3),
        "gsm8k": (0, 1.6),
        "siqa": (0, 1.6),
        "stackmathqa": (0, 10),
    },
    20: {
        "starcoder": (0, 3.5),
        "musicpile" : (0, 2.25),
        "tulu": (0, 3.5),
        "gsm8k": (0, 10),
        "siqa": (0, 10),
        "stackmathqa": (0, 10),
    }
}
XLIM = {
    150: {
        "starcoder": (0, 6),
        "musicpile" : (0, 5),
        "tulu": (0, 3.5),
        "gsm8k": (0, 4),
        "siqa": (0, 5),
        "stackmathqa": (0, 6),
    },
    60: {
        "starcoder": (0, 5.2),
        "musicpile" : (0, 5),
        "tulu": (0, 3.8),
        "gsm8k": (0, 5),
        "siqa": (0, 4.2),
        "stackmathqa": (0, 6),
    },
    20: {
        "starcoder": (0, 6.5),
        "musicpile" : (0, 6.5),
        "tulu": (0, 4.2),
        "gsm8k": (0, 5),
        "siqa": (0, 5),
        "stackmathqa": (0, 10),
    }
}

def make_tradeoff_pareto_optim_token(results):

    thresholds = dict()
    for size in SIZE:
        tokens_list = TOKEN_LIST[size]
        thresholds[size] = dict()
        for cpt_dataset in CPT_DATASET:
            print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            fig, axs = plt.subplots(1, len(tokens_list), figsize=(len(tokens_list) * FIG_WIDTH, 4), sharey=True, sharex=True)
            # Ensure axs is iterable
            if len(tokens_list) == 1:
                axs = [axs]

            
            global_cpt_min = []
            for col, t in enumerate(tokens_list):
                ax = axs[col]

                for optim in OPTIM:
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

                    ax.scatter(dclm_val, cpt_val, label = OPTIM_MAP[optim], marker="o", color=COLOR_MAP[optim])
                    global_cpt_min.append(cpt_min)
                    # ax.scatter(tmp["wd0"]["x"], tmp["wd0"]["y"], label = OPTIM_MAP[optim] + " wd0" , marker="*")
                    # ax.scatter(tmp["bs32"]["x"], tmp["bs32"]["y"], label = OPTIM_MAP[optim] + " wd1e-1", marker="v")
                    # ax.scatter(tmp["bs128"]["x"], tmp["bs128"]["y"], label = OPTIM_MAP[optim] + " bs128", marker="p")
                    if optim == "adamw" and size == 150:
                        for x, y, lr in zip(dclm_val, cpt_val, used_cpt_lrs):
                            ax.text(x, y, f"{lr}", fontsize=8, ha='right', va='bottom', color='black')

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
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[optim])

                ax.set_xlabel("Pretrain Loss", fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(f"Fine-tuning Loss", fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"Tokens / Param = {int(t * 1000 / size)}", fontsize=FONTSIZE["TITLE"])

                ax.legend(fontsize=FONTSIZE["LEGEND"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            cpt_threshold = max(global_cpt_min)
            thresholds[size][cpt_dataset] = cpt_threshold
            for col, t in enumerate(tokens_list):
                ax = axs[col]
                # cpt_threshold = cpt_min * (1 + TRADEOFF_THRESHOLD[size][cpt_dataset])
                
                ax.axhline(cpt_threshold, linestyle='--', color='black', label="FT Threshold")

            fig.suptitle(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["SUP_TITLE"])
            plt.tight_layout() # type: ignore

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_optim_token.png"), bbox_inches='tight')
            plt.close()
        
    with open(os.path.join(RESULTS_DIR, "thresholds_optim_token.json"), "w") as file:
        json.dump(thresholds, file, indent=4)


def make_tradeoff_pareto_optim_compute(results):

    thresholds = dict()
    for size in SIZE:
        tokens_list = TOKEN_LIST[size]
        thresholds[size] = dict()
        for cpt_dataset in CPT_DATASET:
            fig, axs = plt.subplots(1, len(tokens_list)-1, figsize=(len(tokens_list) * FIG_WIDTH, 4), sharey=True, sharex=True)
            # Ensure axs is iterable
            if len(tokens_list) == 1:
                axs = [axs]

            global_cpt_min = []
            for col in range(len(tokens_list)-1):

                ax = axs[col]
                for optim in OPTIM:
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

                    ax.scatter(dclm_val, cpt_val, label = OPTIM_MAP[optim], marker="o", color=COLOR_MAP[optim])
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
                            
                    ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[optim])

                ax.set_xlabel("Pretrain Loss", fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(f"Fine-tuning Loss", fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"Relative FLOPs = {int(tokens_list[col] / min(tokens_list))}", fontsize=FONTSIZE["TITLE"])

                ax.legend(fontsize=FONTSIZE["LEGEND"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            cpt_threshold = max(global_cpt_min)
            thresholds[size][cpt_dataset] = cpt_threshold
            for col in range(len(tokens_list)-1):
                ax = axs[col]
                ax.axhline(cpt_threshold, linestyle='--', color='black', label="FT Threshold")

            fig.suptitle(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["SUP_TITLE"])
            plt.tight_layout(rect=(0, 0, 1, 0.98)) # type: ignore

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_optim_compute.png"), bbox_inches='tight')
            plt.close()

    with open(os.path.join(RESULTS_DIR, "thresholds_optim_compute.json"), "w") as file:
        json.dump(thresholds, file, indent=4)


def make_tradeoff_pareto_lrs(results, anneal_match="token"):

    thresholds = dict()
    optim = "adamw"
    for size in SIZE:
        tokens_list = TOKEN_LIST[size]
        thresholds[size] = dict()
        for cpt_dataset in CPT_DATASET:
            print(f"Plotting {cpt_dataset.capitalize()} for OLMo {size}M")
            fig, axs = plt.subplots(1, len(tokens_list), figsize=(len(tokens_list) * FIG_WIDTH, 4), sharey=True, sharex=True)
            # Ensure axs is iterable
            if len(tokens_list) == 1:
                axs = [axs]

            
            global_cpt_min = []
            for col, t in enumerate(tokens_list):
                ax = axs[col]

                for lrs in LRS:
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

                    ax.scatter(dclm_val, cpt_val, label = LRS_MAP[lrs], marker="o", color=COLOR_MAP[lrs])
                    global_cpt_min.append(cpt_min)
                    if lrs == "wsd_adamw" or lrs == "wsd_sam" and size == 150:
                        for x, y, lr in zip(dclm_val, cpt_val, used_cpt_lrs):
                            ax.text(x, y, f"{lr}", fontsize=8, ha='right', va='bottom', color='black')

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
                                
                        ax.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[lrs])

                ax.set_xlabel("Pretrain Loss", fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(f"Fine-tuning Loss", fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"Tokens / Param = {int(t * 1000 / size)}", fontsize=FONTSIZE["TITLE"])

                ax.legend(fontsize=FONTSIZE["LEGEND"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            cpt_threshold = max(global_cpt_min)
            thresholds[size][cpt_dataset] = cpt_threshold
            for col, t in enumerate(tokens_list):
                ax = axs[col]
                # cpt_threshold = cpt_min * (1 + TRADEOFF_THRESHOLD[size][cpt_dataset])
                
                ax.axhline(cpt_threshold, linestyle='--', color='black', label="FT Threshold")

            fig.suptitle(CPT_DATASET_MAP[cpt_dataset], fontsize=FONTSIZE["SUP_TITLE"])
            plt.tight_layout() # type: ignore

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_lrs_{anneal_match}.png"), bbox_inches='tight')
            plt.close()
        
    with open(os.path.join(RESULTS_DIR, "thresholds_lrs_token.json"), "w") as file:
        json.dump(thresholds, file, indent=4)




def main(args):

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    if args.plot == "optim":
        if args.type == "token":
            make_tradeoff_pareto_optim_token(results)
        elif args.type == "compute":
            make_tradeoff_pareto_optim_compute(results)
    elif args.plot == "lrs":
        make_tradeoff_pareto_lrs(results, anneal_match=args.type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="optim", choices=["optim", "lrs"])
    parser.add_argument("--type", type=str, default="token", choices=["token", "compute"])
    args = parser.parse_args()
    main(args)