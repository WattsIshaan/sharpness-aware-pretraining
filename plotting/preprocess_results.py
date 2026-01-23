import json
import os
import re
from utils.config_globals import RESULTS_DIR, CHECKPOINT_MAP, TASKNAME_MAP

def main():

    results = list()
    raw_results_dir = os.path.join(RESULTS_DIR, "ModelEvaluation")

    for file in os.listdir(raw_results_dir):

        fname = os.path.join(raw_results_dir, file)
        with open(fname, 'r') as f:
            data = json.load(f)

        fname = fname.removesuffix("-eval.json")
        
        # MODEL SIZE
        size_match = re.search(r'OLMo2-(\d+)m-', fname)
        if size_match:
            size = int(size_match.group(1))
        else:
            size = None  # Could be "unknown" if not found


        # Check if this is a 'perturbed' run (look for "_perturbed_<value>" in fname)
        perturbation = None
        # Support both _perturbed_1e-2 and _perturbed_1_50e-2
        perturbed_match = re.search(r'_perturbed_([0-9_\.eE\-\+]+)', fname)
        if perturbed_match:
            pert_val_str = perturbed_match.group(1)
            # Handle values like "1_50e-2": convert to float by replacing first "_" with "."
            if "_" in pert_val_str:
                # Try to parse forms like "1_50e-2"
                # This is to match prompts like "OLMo-20m-tk300B-adamw-lr3e-4-wd1e-1-bs256-anneal-ckpt15000_perturbed_1_50e-2-eval.json"
                pert_val_str_parts = pert_val_str.split("_", 1)
                if len(pert_val_str_parts) == 2:
                    new_val_str = pert_val_str_parts[0] + "." + pert_val_str_parts[1]
                else:
                    new_val_str = pert_val_str.replace("_", ".")   # fallback
                try:
                    perturbation = float(new_val_str)
                except Exception:
                    raise ValueError(f"Could not convert perturbation value: {pert_val_str!r} became {new_val_str!r}")
            else:
                try:
                    perturbation = float(pert_val_str)
                except Exception:
                    raise ValueError(f"Could not convert perturbation value: {pert_val_str!r}")

        is_cpt = 'CPT' in fname
        if perturbation is not None:
            run_type = "perturbed"
        elif is_cpt:
            run_type = "cpt"
        elif "quant" in fname:
            run_type = "quant"
        else:
            run_type = "pretrain"

        quant_bit = None
        if "quant" in fname:
            # Parse quantization bit value if present (e.g., '-quant-8bit')
            quant_match = re.search(r'-quant-(\d+)bit', fname)
            if quant_match:
                try:
                    quant_bit = int(quant_match.group(1))
                except Exception:
                    raise ValueError(f"Could not parse quantization bit value from filename: {fname!r}")

        # If "anneal" is present, get ckpt<> value as token, else get tk<>B
        anneal_steps, anneal_optim, anneal_percent, anneal_optim = None, None, None, None
        if "anneal" in fname:
            pt_lrs = "wsd"
            # Check for 'steps' and extract the int value
            steps_match = re.search(r'steps(\d+)', fname)
            percent_match = re.search(r'percent(\d+)', fname)
            if steps_match:
                anneal_steps = int(steps_match.group(1))
            if percent_match:
                anneal_percent = int(percent_match.group(1))
            # Try to find 'ckpt<digits>' (e.g. ckpt55000)
            ckpt_match = re.search(r'ckpt(\d+)', fname)
            if ckpt_match:
                try:
                    token = CHECKPOINT_MAP[size][int(ckpt_match.group(1))] # type: ignore
                except:
                    print(f"Skipping ckpt {ckpt_match.group(1)} ", fname)
                    continue
            else:
                raise ValueError("Unable to find ckpt value in filename with anneal: %s" % fname)
            
            # if size == 60 and run_type == "pretrain" and anneal_percent == 5:
            #     print(token, run_type, anneal_percent, anneal_steps)
            #     print(fname)
            # Now parse the optimizer following '-anneal-', which is either 'sam' or 'adamw'
            if "anneal-sam" in fname or "anneal2" in fname:
                anneal_optim = "sam"
            else:
                anneal_optim = "adamw"

        else:
            token_match = re.search(r'tk(\d+)B', fname)
            token = token_match.group(1) if token_match else "unknown"
            pt_lrs = "cosine"
            if token != "unknown":
                try:
                    token = int(token)
                except Exception:
                    raise ValueError("Unable to convert tk token to Int: %s" % fname)
            else:
                raise ValueError("Unable to Parse tk<B> for token: %s" % fname)

        # Get the optimizer (can be sam, adamw, lionw, sam_adamw, sam_sgd)
        optimizer_match = re.search(r'tk\d+B-([a-zA-Z0-9_]+)-lr', fname)
        optimizer_raw = optimizer_match.group(1) if optimizer_match else "unknown"

        # Handle sam_adamw and sam_sgd specially
        if optimizer_raw in ("sam_adamw", "sam_sgd"):
            optimizer = "sam"
            base_optimizer = optimizer_raw.split("_")[1]

            # Properly extract rho after "-rho", e.g.: "...-bs256-rho1e-1-"
            rho_match = re.search(r'-rho([0-9eE\+\-\.]+)', fname)
            if not rho_match:
                # fallback: some files may omit '-rho', to maintain old behavior if necessary
                rho_match = re.search(r'-bs256-([0-9eE\+\-\.]+)-', fname)
            rho = rho_match.group(1) if rho_match else "unknown"

            if rho != "unknown":
                rho_cleaned = re.sub(r'[^0-9eE\+\-\.]+$', '', rho)
                rho_cleaned = rho_cleaned.rstrip('-')
                try:
                    rho_val = float(rho_cleaned)
                except Exception:
                    raise ValueError(f"Unable to convert rho to float: {rho_cleaned!r}")
            else:
                raise ValueError("Unable to Parse rho")
        else:
            optimizer = optimizer_raw
            base_optimizer = None
            rho_val = None

        # Always get pretrain_lr (after the optimizer)
        pretrain_lr_match = re.search(rf'{optimizer_raw}-lr([0-9eE\+\-\.]+)', fname)
        pretrain_lr = pretrain_lr_match.group(1) if pretrain_lr_match else "unknown"
        if pretrain_lr != "unknown":
            pretrain_lr_cleaned = re.sub(r'[^0-9eE\+\-\.]+$', '', pretrain_lr)
            pretrain_lr_cleaned = pretrain_lr_cleaned.rstrip('-')
            try:
                pretrain_lr = float(pretrain_lr_cleaned)
            except Exception:
                raise ValueError(f"Unable to convert to Float: {pretrain_lr_cleaned!r}")
        else:
            raise ValueError("Unable to Parse pretrain_lr in %s" % fname)

        # Always get pretrain_wd (right after pretrain_lr)
        pretrain_wd_match = re.search(rf'{optimizer_raw}-lr[0-9eE\+\-\.]+-wd([0-9eE\+\-\.]+)', fname)
        pretrain_wd = pretrain_wd_match.group(1) if pretrain_wd_match else "unknown"
        if pretrain_wd != "unknown":
            pretrain_wd_cleaned = re.sub(r'[^0-9eE\+\-\.]+$', '', pretrain_wd)
            pretrain_wd_cleaned = pretrain_wd_cleaned.rstrip('-')
            try:
                pretrain_wd = float(pretrain_wd_cleaned)
            except Exception:
                raise ValueError(f"Unable to convert to Float: {pretrain_wd_cleaned!r}")
        else:
            raise ValueError("Unable to Parse pretrain_wd in %s" % fname)

        # Default cpt values
        cpt_lr, cpt_wd, cpt_dataset, cpt_tokens, cpt_bs = None, None, None, None, None
        if is_cpt:
            # Extract cpt_dataset and cpt_tokens (tk{5,10,20}M). Default cpt_tokens is 10 if not present
            # Allow '-' in dataset names, e.g., open-platypus
            cpt_dataset_match = re.search(r'CPT-([a-zA-Z0-9_\-]+)-tk([0-9]+)M-lr', fname)
            if cpt_dataset_match:
                cpt_dataset = cpt_dataset_match.group(1)
                cpt_tokens = int(cpt_dataset_match.group(2))
            else:
                # Fallback to cpt_dataset only (for compatibility)
                cpt_dataset_match = re.search(r'CPT-([a-zA-Z0-9_\-]+)-lr', fname)
                cpt_dataset = cpt_dataset_match.group(1) if cpt_dataset_match else "unknown"
                cpt_tokens = 10  # default

            # Find correct learning rate key for CPT dataset with tk tokens present
            cpt_lr_match = re.search(rf'CPT-{cpt_dataset}-tk{cpt_tokens}M-lr([0-9eE\+\-\.]+)', fname)
            if not cpt_lr_match:
                # Fallback in case format is old (no tk tokens)
                cpt_lr_match = re.search(rf'CPT-{cpt_dataset}-lr([0-9eE\+\-\.]+)', fname)
            cpt_lr = cpt_lr_match.group(1) if cpt_lr_match else "unknown"
            if cpt_lr != "unknown":
                cpt_lr_cleaned = re.sub(r'[^0-9eE\+\-\.]+$', '', cpt_lr)
                cpt_lr_cleaned = cpt_lr_cleaned.rstrip('-')
                try:
                    cpt_lr = float(cpt_lr_cleaned)
                except Exception:
                    raise ValueError(f"Unable to convert to Float: {cpt_lr_cleaned!r}")
            else:
                raise ValueError("Unable to Parse, %s" % fname)

            # Try to match wd including a possible "0" (zero) as value (OLMo2-60m-tk12B-adamw-lr1e-3-wd1e-1-bs256-CPT-starcoder-tk10M-lr4.00e-4-wd0-bs64-eval.json etc)
            cpt_wd_match = re.search(
                rf'CPT-{cpt_dataset}-tk{cpt_tokens}M-lr[0-9eE\+\-\.]+-wd([0-9eE\+\-\.]*0*|\d+)', fname)
            if not cpt_wd_match:
                # Fallback in case format is old (no tk tokens)
                cpt_wd_match = re.search(
                    rf'CPT-{cpt_dataset}-lr[0-9eE\+\-\.]+-wd([0-9eE\+\-\.]*0*|\d+)', fname)
            cpt_wd = cpt_wd_match.group(1) if cpt_wd_match else "unknown"
            if cpt_wd != "unknown":
                cpt_wd_cleaned = re.sub(r'[^0-9eE\+\-\.]+$', '', cpt_wd)
                cpt_wd_cleaned = cpt_wd_cleaned.rstrip('-')
                try:
                    cpt_wd = float(cpt_wd_cleaned)
                except Exception:
                    raise ValueError(f"Unable to convert to Float: {cpt_wd_cleaned!r}")
            else:
                raise ValueError("Unable to Parse")
            
            # Try to parse with tk tokens
            cpt_bs_match = re.search(
                rf'CPT-{cpt_dataset}-tk{cpt_tokens}M-lr[0-9eE\+\-\.]+-wd[0-9eE\+\-\.]*-bs([0-9]+)',
                fname
            )
            if not cpt_bs_match:
                # Fallback: parse if tk tokens not present in fname pattern
                cpt_bs_match = re.search(
                    rf'CPT-{cpt_dataset}-lr[0-9eE\+\-\.]+-wd[0-9eE\+\-\.]*-bs([0-9]+)',
                    fname
                )
            if cpt_bs_match:
                cpt_bs = int(cpt_bs_match.group(1))
            else:
                cpt_bs = None  # Not found; optional


        if "hf" in fname:
            model_type = "hf"
        else:
            model_type = "olmo"

        run_info = {
            "run_type": run_type,
            "model_type": model_type,
            "token": token,
            "optimizer": optimizer,
            "pretrain_lr": pretrain_lr,
            "pretrain_wd": pretrain_wd,
            "pretrain_lrs": pt_lrs,
            # "c4_val": data[TASKNAME_MAP["c4_val"]],
            "dclm_val": data[TASKNAME_MAP["dclm_val"]],
            "size": size,
        }
        if run_info["run_type"] == "pretrain":
            run_info["dclm_train"] = data.get(TASKNAME_MAP["dclm_train"], None)
        if perturbation is not None:
            run_info["perturbation"] = perturbation
        if run_info["run_type"] == "quant":
            run_info["quant_bit"] = quant_bit
        if base_optimizer is not None:
            run_info["base_optimizer"] = base_optimizer
        if rho_val is not None:
            run_info["rho"] = rho_val
        if is_cpt:
            run_info["cpt_lr"] = cpt_lr
            run_info["cpt_wd"] = cpt_wd
            run_info["cpt_dataset"] = cpt_dataset
            run_info["cpt_tokens"] = cpt_tokens
            run_info["cpt_bs"] = cpt_bs
            run_info[f"{cpt_dataset}_val"] = data[TASKNAME_MAP[f"{cpt_dataset}_val"]]
        if pt_lrs == "wsd":
            run_info["anneal_steps"] = anneal_steps
            run_info["anneal_percent"] = anneal_percent
            run_info["anneal_optim"] = anneal_optim
        
        results.append(run_info)

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "w") as file:
        json.dump(results, file, indent=2)

if __name__ == "__main__":
    main()