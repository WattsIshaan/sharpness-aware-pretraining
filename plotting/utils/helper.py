from utils.config_globals import TOKEN_LIST, PT_LR, PERTURBATIONS, VERBOSE, BITS
from utils.plotting_globals import OPTIM_MAP

def get_run_info(
    results, 
    size, 
    optim, 
    cpt_dataset=None, 
    cpt_tokens=10,
    rho=5e-2, 
    perturb=False, 
    quantized=False, 
    pt_lr: float | str = "adapt",
    anneal=False, 
    anneal_steps=None, 
    anneal_percent=None, 
    model_type="olmo", 
    anneal_optim="adamw",
    anneal_match="token"
):

    run_info = dict()
    run_info["pretrain"] = dict()
    
    tokens_list = TOKEN_LIST[size]
    all_runs = [r for r in results if r.get("token") in tokens_list and r.get("size") == size and r.get("optimizer") == optim and r.get("model_type") == model_type]

    if pt_lr == "adapt":
        all_runs = [r for r in all_runs if r.get("pretrain_lr") == PT_LR[size][r["token"]]]
    else:
        all_runs = [r for r in all_runs if r.get("pretrain_lr") == pt_lr]

    if anneal:
        assert anneal_steps is not None or anneal_percent is not None 
        all_runs = [r for r in all_runs if r.get("pretrain_lrs") == "wsd" and r.get("anneal_optim") == anneal_optim and r.get("anneal_match") == (anneal_match or "both")]
        if anneal_steps is not None:
            all_runs = [r for r in all_runs if r.get("anneal_steps") == anneal_steps]
        if anneal_percent is not None:
            all_runs = [r for r in all_runs if r.get("anneal_percent") == anneal_percent]  
    else:
        all_runs = [r for r in all_runs if r.get("pretrain_lrs") == "cosine"]

    if optim == "sam":
        all_runs = [r for r in all_runs if r.get("rho") == rho]

    if len(all_runs) == 0:
        return None

    pretrain_runs = [r for r in all_runs if r.get("run_type") == "pretrain"]
    for t in tokens_list:
        run = [r for r in pretrain_runs if r.get("token") == t]
        if len(run) == 0:
            if VERBOSE:
                print(f"Missing Run: Optim {OPTIM_MAP[optim]} | Token {t}B | Size {size}M")
            continue
        assert len(run) == 1, print(run)
        run = run[0]

        run_info["pretrain"][t] = {
            "token" : t,
            "multiplier" : (t * 1000) / size,
            # "c4": run["c4_val"],
            "dclm_val": run["dclm_val"],
            "dclm_train": run["dclm_train"]
        }

    # print(pretrain_runs)

    if perturb:
        run_info["perturbed"] = dict()
        perturbed_runs = [r for r in all_runs if r.get("run_type") == "perturbed"]

        for perturbation in PERTURBATIONS:
            run_info["perturbed"][perturbation] = dict()
            run_perturbed = [r for r in perturbed_runs if r.get("perturbation") == perturbation]

            for t in tokens_list:
                run_pt = [r for r in pretrain_runs if r.get("token") == t]
                if len(run_pt) == 0:
                    continue
                assert len(run_pt) == 1, print(run_pt)
                run_pt = run_pt[0]
                run_pert = [r for r in run_perturbed if r.get("token") == t]
                if len(run_pert) == 0:
                    continue
                assert len(run_pert) == 1, print(run_pert)
                run_pert = run_pert[0]

                run_info["perturbed"][perturbation][t] = {
                    "token" : t,
                    "multiplier" : (t * 1000) / size,
                    "perturbed" : perturbation,
                    "dclm_perturbed" : run_pert["dclm_val"],
                    "dclm_val" : run_pt["dclm_val"]
                }
    
    if quantized:
        run_info["quantized"] = dict()
        quantized_runs = [r for r in all_runs if r.get("run_type") == "quant"]

        for bit in BITS:
            run_info["quantized"][bit] = dict()
            run_quantized = [r for r in quantized_runs if r.get("quant_bit") == bit]

            for t in tokens_list:
                run_quant = [r for r in run_quantized if r.get("token") == t]

                if len(run_quant) == 0:
                    continue
                assert len(run_quant) == 1, print(run_quant)
                run_quant = run_quant[0]

                run_info["quantized"][bit][t] = {
                    "token" : t,
                    "multiplier" : (t * 1000) / size,
                    "quant_bit" : bit,
                    "dclm_quant" : run_quant["dclm_val"],
                }

    if cpt_dataset is not None:

        run_info["cpt"] = dict()
        cpt_runs = [r for r in all_runs if r.get("run_type") == "cpt" and r.get("cpt_dataset") == cpt_dataset and r.get("cpt_tokens") == cpt_tokens]
        cpt_lrs = sorted(list(set([r.get("cpt_lr") for r in cpt_runs])))
        cpt_wds = sorted(list(set([r.get("cpt_wd") for r in cpt_runs])))
        cpt_bss = sorted(list(set([r.get("cpt_bs") for r in cpt_runs])))

        for cpt_lr in cpt_lrs:
            run_info["cpt"][cpt_lr] = dict()
            for cpt_wd in cpt_wds:
                run_info["cpt"][cpt_lr][cpt_wd] = dict()
                for cpt_bs in cpt_bss:
                    run_info["cpt"][cpt_lr][cpt_wd][cpt_bs] = dict()
                     
                    for t in tokens_list:
                        run = [r for r in cpt_runs if r.get("cpt_lr") == cpt_lr and r.get("token") == t and r.get("cpt_wd") == cpt_wd and r.get("cpt_bs") == cpt_bs]
                        if len(run) == 0:
                            if VERBOSE:
                                print(f"Missing Run: Optim {OPTIM_MAP[optim]} | Token {t}B | Size {size}M | CPT dataset {cpt_dataset} | CPT LR {cpt_lr} | CPT WD {cpt_wd} | CPT BS {cpt_bs}")
                            continue
                        
                        if run[0].get("cpt_tokens") != 10:
                            assert len(run) == 1, print(run)
                        run = run[0]

                        run_info["cpt"][cpt_lr][cpt_wd][cpt_bs][t] = {
                            "token": t,
                            "multiplier" : (t * 1000) / size,
                            "cpt_lr": cpt_lr,
                            "cpt_bs": cpt_bs,
                            "cpt_wd": cpt_wd,
                            "dclm_val": run["dclm_val"],
                            cpt_dataset: run[f"{cpt_dataset}_val"]
                        }

    return run_info

