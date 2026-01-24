import matplotlib.pyplot as plt
import os
import json
from scipy.spatial import ConvexHull
import numpy as np

from utils.config_globals import SIZE, CPT_DATASET, OPTIM, TOKEN_LIST, RESULTS_DIR
from utils.plotting_globals import OPTIM_MAP, FONTSIZE, FIG_WIDTH, COLOR_MAP, LRS_MAP
from utils.helper import get_run_info

YLIM = {
    150: {
        "starcoder": (0, 2.5),
        "musicpile" : (0, 1.55),
        "tulu": (0, 3),
        "gsm8k": (0, 10),
        "alpaca": (0, 10),
        "siqa": (0, 10),
        "open-platypus": (0, 10),
        "stackmathqa": (0, 10),
        "helpsteer": (0, 10),
    },
    60: {
        "starcoder": (0, 2.8),
        "musicpile" : (0, 1.7),
        "tulu": (0, 3),
        "gsm8k": (0, 1.6),
        "alpaca": (0, 10),
        "siqa": (0, 1.6),
        "open-platypus": (0, 10),
        "stackmathqa": (0, 10),
        "helpsteer": (0, 10),
    },
    20: {
        "starcoder": (0, 3.5),
        "musicpile" : (0, 2.25),
        "tulu": (0, 3.5),
        "gsm8k": (0, 10),
        "alpaca": (0, 10),
        "siqa": (0, 10),
        "open-platypus": (0, 10),
        "stackmathqa": (0, 10),
        "helpsteer": (0, 10),
    }
}
XLIM = {
    150: {
        "starcoder": (0, 6),
        "musicpile" : (0, 5),
        "tulu": (0, 3.5),
        "gsm8k": (0, 5),
        "alpaca": (0, 10),
        "siqa": (0, 5),
        "open-platypus": (0, 10),
        "stackmathqa": (0, 10),
        "helpsteer": (0, 10),
    },
    60: {
        "starcoder": (0, 5.2),
        "musicpile" : (0, 5),
        "tulu": (0, 3.8),
        "gsm8k": (0, 5),
        "alpaca": (0, 10),
        "siqa": (0, 5),
        "open-platypus": (0, 4),
        "stackmathqa": (0, 6),
        "helpsteer": (0, 4),
    },
    20: {
        "starcoder": (0, 6.5),
        "musicpile" : (0, 6.5),
        "tulu": (0, 4.2),
        "gsm8k": (0, 5),
        "alpaca": (0, 10),
        "siqa": (0, 5),
        "open-platypus": (0, 10),
        "stackmathqa": (0, 10),
        "helpsteer": (0, 10),
    }
}

def make_tradeoff_pareto_token_matched(results):

    for size in SIZE:
        tokens_list = TOKEN_LIST[size]
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

                    tmp = {
                        "bs32": {
                            "x": [],
                            "y": []
                        },
                        "bs128": {
                            "x": [],
                            "y": []
                        },
                        "wd0": {
                            "x": [],
                            "y": []
                        },
                    }
                    for cpt_lr in cpt_lrs:
                        cpt_wds = sorted(run_info["cpt"][cpt_lr].keys())
                        for cpt_wd in cpt_wds:
                            cpt_bss = sorted(run_info["cpt"][cpt_lr][cpt_wd].keys())
                            for cpt_bs in cpt_bss:
                                try:
                                    x_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t]["dclm_val"]
                                    y_val = run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t][cpt_dataset]

                                    if x_val >= XLIM[size][cpt_dataset][0] and x_val <= XLIM[size][cpt_dataset][1] and y_val >= YLIM[size][cpt_dataset][0] and y_val <= YLIM[size][cpt_dataset][1]:
                                        if cpt_wd == 1e-1:
                                            tmp["wd0"]["x"].append(x_val)
                                            tmp["wd0"]["y"].append(y_val)
                                        elif cpt_wd == 0:
                                            tmp["bs32"]["x"].append(x_val)
                                            tmp["bs32"]["y"].append(y_val)
                                        # if cpt_bs == 128:
                                        #     tmp["bs128"]["x"].append(x_val)
                                        #     tmp["bs128"]["y"].append(y_val)


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
                    if optim == "sam":
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

                ax.set_xlabel("DCLM Val Loss", fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(f"{cpt_dataset.capitalize()} Val Loss", fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"Tokens / Param = {int(t * 1000 / size)}", fontsize=FONTSIZE["TITLE"])

                ax.legend(fontsize=FONTSIZE["LEGEND"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            for col, t in enumerate(tokens_list):
                ax = axs[col]
                # cpt_threshold = cpt_min * (1 + TRADEOFF_THRESHOLD[size][cpt_dataset])
                cpt_threshold = max(global_cpt_min)
                ax.axhline(cpt_threshold, linestyle='--', color='black', label="FT Threshold")

            fig.suptitle(f"OLMo-{size}M | {cpt_dataset.capitalize()} | FT Loss v/s PT Loss (Token Matched)", fontsize=FONTSIZE["SUP_TITLE"])
            plt.tight_layout(rect=(0, 0, 1, 0.96)) # type: ignore

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_token.png"), bbox_inches='tight')
            plt.close()



def make_tradeoff_pareto_compute_matched(results):

    for size in SIZE:
        tokens_list = TOKEN_LIST[size]
        
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

                    # tmp = {
                    #     "bs32": {
                    #         "x": [],
                    #         "y": []
                    #     },
                    #     "bs128": {
                    #         "x": [],
                    #         "y": []
                    #     },
                    #     "wd0": {
                    #         "x": [],
                    #         "y": []
                    #     },
                    # }
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
                                        # if cpt_wd == 0 and cpt_bs == 64:
                                        #     tmp["wd0"]["x"].append(x_val)
                                        #     tmp["wd0"]["y"].append(y_val)
                                        # if cpt_bs == 32:
                                        #     tmp["bs32"]["x"].append(x_val)
                                        #     tmp["bs32"]["y"].append(y_val)
                                        # if cpt_bs == 128:
                                        #     tmp["bs128"]["x"].append(x_val)
                                        #     tmp["bs128"]["y"].append(y_val)


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

                ax.set_xlabel("DCLM Val Loss", fontsize=FONTSIZE["AXIS"])
                if col == 0:
                    ax.set_ylabel(f"{cpt_dataset.capitalize()} Val Loss", fontsize=FONTSIZE["AXIS"])
                ax.set_title(f"Relative FLOPs = {int(tokens_list[col] / min(tokens_list))}", fontsize=FONTSIZE["TITLE"])

                ax.legend(fontsize=FONTSIZE["LEGEND"])
                ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
                ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
                ax.grid(True)

            for col in range(len(tokens_list)-1):
                ax = axs[col]
                cpt_threshold = max(global_cpt_min)
                ax.axhline(cpt_threshold, linestyle='--', color='black', label="FT Threshold")

            fig.suptitle(f"OLMo-{size}M | {cpt_dataset.capitalize()} | FT Loss v/s PT Loss (Compute Matched)", fontsize=FONTSIZE["SUP_TITLE"])
            plt.tight_layout(rect=(0, 0, 1, 0.96)) # type: ignore

            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_compute.png"), bbox_inches='tight')
            plt.close()


def make_tradeoff_pareto_lrs(results):

    optim = "adamw"
    for size in SIZE:
        token_budget = TOKEN_LIST[size][-1]
        for cpt_dataset in CPT_DATASET:

            plt.figure(figsize=(FIG_WIDTH, 4))  # <-- fix plot size for each plot (width, height)

            for lrs in ["wsd_adamw", "wsd_sam", "cosine"]:
                if lrs == "cosine":
                    run_info = get_run_info(results, size, optim, cpt_dataset, anneal=False)
                else:
                    anneal_optim = lrs.split("_")[1]
                    run_info = get_run_info(results, size, optim, cpt_dataset, anneal=True, anneal_percent=10, anneal_optim=anneal_optim)

                if run_info is None:
                    continue

                cpt_lrs = sorted(run_info["cpt"].keys())
                dclm_val = []
                cpt_val = []
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
                                    # cpt_min = min(y_val, cpt_min)
                            except:
                                continue

                plt.scatter(dclm_val, cpt_val, label=LRS_MAP[lrs], marker="o", color=COLOR_MAP[lrs])

                points = np.empty((len(dclm_val), 2))
                points[:, 0] = dclm_val
                points[:, 1] = cpt_val
                hull = ConvexHull(points)
                h = np.append(hull.vertices, hull.vertices[0])
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

                plt.plot(new_points[:,0], new_points[:,1], color=COLOR_MAP[lrs])

            plt.xlabel("DCLM Val Loss", fontsize=FONTSIZE["AXIS"])
            plt.ylabel(f"{cpt_dataset.capitalize()} Val Loss", fontsize=FONTSIZE["AXIS"])
            plt.title(f"FT Loss v/s PT Loss across Learning Rate Schedulers", fontsize=FONTSIZE["TITLE"])

            plt.legend(fontsize=FONTSIZE["LEGEND"])
            plt.tick_params(axis='both', which='major', labelsize=FONTSIZE["TICKS"])
            plt.tick_params(axis='both', which='minor', labelsize=FONTSIZE["TICKS"])
            plt.grid(True)

            plt.tight_layout(rect=(0, 0, 1, 0.96)) # type: ignore
            os.makedirs(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m"), exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, f"plots/pareto/{size}m/{cpt_dataset}_lrs.png"), bbox_inches='tight')
            plt.close()



def main():

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "r") as file:
        results = json.load(file)

    make_tradeoff_pareto_token_matched(results)
    # make_tradeoff_pareto_compute_matched(results)
    # make_tradeoff_pareto_lrs(results)



if __name__ == "__main__":
    main()