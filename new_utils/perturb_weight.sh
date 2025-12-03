#!/usr/bin/env bash

#SBATCH --job-name=perturb_weight
#SBATCH --output=logs/perturb_weight_%j.out
#SBATCH --error=logs/perturb_weight_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 
#SBATCH --requeue
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu_qos
#SBATCH --account=aditirag

source ~/miniconda3/etc/profile.d/conda.sh
conda activate forgetting


# List of sigma values
# SIGMAS=(0.009 0.0095 0.013 0.015 0.017 0.022 0.025 0.02 0.01)
SIGMAS=(0.03 0.04 0.05 0.075 0.1)
BASE_MODEL_PATH="gs://cmu-gpucloud-iwatts/outputs/large-scale-experiments/AnnealedModel"
OPTIM="adamw"

CKPT_STEPS=(670000 40000 85000 165000 335000) #60m
PT_TOKEN=400
MODEL_SIZE=60
# CKPT_STEPS=(15000 30000 55000 110000 220000 445000 890000) #20m
# PT_TOKEN=300
# MODEL_SIZE=20


# Loop over sigmas and run the script
for ckpt_step in "${CKPT_STEPS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        if [ "$OPTIM" == "sam_adamw" ]; then
            MODEL_NAME=OLMo-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr3e-4-wd1e-1-bs256-rho5e-2-anneal-ckpt${ckpt_step}
        else
            MODEL_NAME=OLMo-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr3e-4-wd1e-1-bs256-anneal-ckpt${ckpt_step}
        fi
        echo "Running with sigma=${sigma}"
        python /home/iwatts/catastrophic-forgetting/new_utils/perturb_weights.py \
            --gcs_dir "${BASE_MODEL_PATH}" \
            --model_name "${MODEL_NAME}" \
            --output_gcs_dir "gs://cmu-gpucloud-iwatts/outputs/large-scale-experiments/PerturbedModel" \
            --sigma "${sigma}"
    done
done

