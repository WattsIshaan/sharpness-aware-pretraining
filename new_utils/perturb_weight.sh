#!/usr/bin/env bash

#SBATCH --job-name=perturb_weight
#SBATCH --output=logs/perturb_weight_%j.out
#SBATCH --error=logs/perturb_weight_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 
#SBATCH --requeue
#SBATCH --partition=preempt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate forgetting2


# List of sigma values
# SIGMAS=(0.009 0.013 0.017 0.02 0.025 0.03 0.05 0.075 0.1)
SIGMAS=(0.1)
PROJECT="60m-experiments"
BASE_MODEL_PATH="gs://cmu-gpucloud-iwatts/outputs/${PROJECT}/PretrainedModel"
OPTIM="adamw"

# PT_TOKEN=(4 8 16)
# PT_TOKEN=(32 64)
PT_LR="3e-4"
MODEL_SIZE=60


# Loop over sigmas and run the script
# for pt_token in "${PT_TOKEN[@]}"; do
#     for sigma in "${SIGMAS[@]}"; do
#         if [ "$OPTIM" == "sam_adamw" ]; then
#             MODEL_NAME=OLMo2-${MODEL_SIZE}m-tk${pt_token}B-${OPTIM}-lr${PT_LR}-wd1e-1-bs256-rho5e-2
#         else
#             MODEL_NAME=OLMo2-${MODEL_SIZE}m-tk${pt_token}B-${OPTIM}-lr${PT_LR}-wd1e-1-bs256
#         fi
#         echo "Running with sigma=${sigma}"
#         python /home/iwatts/catastrophic-forgetting/new_utils/perturb_weights.py \
#             --gcs_dir "${BASE_MODEL_PATH}" \
#             --model_name "${MODEL_NAME}" \
#             --output_gcs_dir "gs://cmu-gpucloud-iwatts/outputs/${PROJECT}/PerturbedModel" \
#             --sigma "${sigma}"
#     done
# done

CKPT_STEPS=(45000 90000 180000 365000 730000) #60m
# CKPT_STEPS=(45000 90000 175000 350000 700000)
# CKPT_STEPS=(35000 75000 155000 305000 610000)
# CKPT_STEPS=(670000 40000 85000 165000 335000)
# # ANNEAL_STEPS=(1000 2000 4000)
CKPT_STEPS=(670000)
ANNEAL_PERCENT=(5 10)
# ANNEAL_STEPS=(1000)
BASE_MODEL_PATH="gs://cmu-gpucloud-iwatts/outputs/${PROJECT}/AnnealedModel"
PT_TOKEN=200

# Loop over sigmas and run the script
for ckpt_step in "${CKPT_STEPS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        # for anneal_percent in "${ANNEAL_PERCENT[@]}"; do
        for anneal_steps in "${ANNEAL_STEPS[@]}"; do

            if [ "$OPTIM" == "sam_adamw" ]; then
                # MODEL_NAME=OLMo2-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr${PT_LR}-wd1e-1-bs256-rho5e-2-anneal-ckpt${ckpt_step}-percent${anneal_percent}
                MODEL_NAME=OLMo2-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr${PT_LR}-wd1e-1-bs256-rho5e-2-anneal-ckpt${ckpt_step}-steps${anneal_steps}
            else
                # MODEL_NAME=OLMo2-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr${PT_LR}-wd1e-1-bs256-anneal-ckpt${ckpt_step}-percent${anneal_percent}
                MODEL_NAME=OLMo2-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr${PT_LR}-wd1e-1-bs256-anneal2-ckpt${ckpt_step}-steps${anneal_steps}
            fi
            echo "Running with sigma=${sigma}"
            python /home/iwatts/catastrophic-forgetting/new_utils/perturb_weights.py \
                --gcs_dir "${BASE_MODEL_PATH}" \
                --model_name "${MODEL_NAME}" \
                --output_gcs_dir "gs://cmu-gpucloud-iwatts/outputs/${PROJECT}/PerturbedModel" \
                --sigma "${sigma}"
        done
    done
done

