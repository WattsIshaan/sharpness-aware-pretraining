#!/usr/bin/env bash

#SBATCH --job-name=evaluate_perturbed
#SBATCH --output=logs/evaluate_perturbed_%j.out
#SBATCH --error=logs/evaluate_perturbed_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --partition=general

source ~/miniconda3/etc/profile.d/conda.sh
conda activate forgetting2


# List of sigma values (must match perturb_weight.sh)
# SIGMAS=(0.009 0.013 0.017 0.02 0.025 0.03 0.05 0.075 0.1)
SIGMAS=(0.1)
PROJECT="60m-experiments"
BASE_MODEL_PATH="gs://cmu-gpucloud-iwatts/outputs/${PROJECT}/PerturbedModel"
OPTIM="adamw"
MODEL_SIZE=60

# PT_TOKEN=(15 30)
# PT_TOKEN=(60 120)
# PT_TOKEN=(4 8 16)
# PT_LR="3e-3"
# PT_TOKEN=(32 64)
# PT_LR="6e-4"

# Function to format sigma for directory name (matches perturb_weights.py)
format_sigma() {
    local sigma=$1
    # Format as scientific notation and replace dots with underscores
    printf "%.2e" "$sigma" | sed 's/e-0/e-/g' | sed 's/e+0/e+/g' | sed 's/\./_/g'
}

# Loop over checkpoint steps and sigmas
# for pt_token in "${PT_TOKEN[@]}"; do
#     for sigma in "${SIGMAS[@]}"; do
#         if [ "$OPTIM" == "sam_adamw" ]; then
#             # MODEL_NAME=OLMo-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr3e-4-wd1e-1-bs256-rho5e-2-anneal-ckpt${ckpt_step}
#             MODEL_NAME=OLMo2-${MODEL_SIZE}m-tk${pt_token}B-${OPTIM}-lr${PT_LR}-wd1e-1-bs256-rho5e-2
#         else
#             # MODEL_NAME=OLMo-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr3e-4-wd1e-1-bs256-anneal-ckpt${ckpt_step}
#             MODEL_NAME=OLMo2-${MODEL_SIZE}m-tk${pt_token}B-${OPTIM}-lr${PT_LR}-wd1e-1-bs256
#         fi
#         SIGMA_STR=$(format_sigma "$sigma")
#         PERTURBED_MODEL_NAME="${MODEL_NAME}_perturbed_${SIGMA_STR}"
        
#         echo "Evaluating model: ${PERTURBED_MODEL_NAME}"
#         echo "  Checkpoint step: ${ckpt_step}"
#         echo "  Sigma: ${sigma}"
        
#         python /home/iwatts/catastrophic-forgetting/new_utils/evaluate_perturbed.py \
#             --gcs_dir "${BASE_MODEL_PATH}" \
#             --model_name "${PERTURBED_MODEL_NAME}" \
#             --output_gcs_dir "gs://cmu-gpucloud-iwatts/outputs/${PROJECT}/ModelEvaluation" \
#             --dtype "uint32"
        
#         echo "✅ Completed evaluation for ${PERTURBED_MODEL_NAME}"
#         echo ""
#     done
# done

CKPT_STEPS=(45000 90000 180000 365000 730000) #60m
# CKPT_STEPS=(45000) #60m
PT_LR="3e-4"
# # CKPT_STEPS=(35000 75000 155000 305000 610000)
# # CKPT_STEPS=(45000 90000 175000 350000 700000)
# # CKPT_STEPS=(670000 40000 85000 165000 335000)
# CKPT_STEPS=(45000)
# # ANNEAL_STEPS=(1000 2000 4000)
# # ANNEAL_PERCENT=(5 10 20)
ANNEAL_STEPS=(1000)
PT_TOKEN=(200)

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
            SIGMA_STR=$(format_sigma "$sigma")
            PERTURBED_MODEL_NAME="${MODEL_NAME}_perturbed_${SIGMA_STR}"
            
            echo "Evaluating model: ${PERTURBED_MODEL_NAME}"
            echo "  Checkpoint step: ${ckpt_step}"
            echo "  Anneal step: ${anneal_steps}"
            # echo "  Anneal percent: ${anneal_percent}"
            echo "  Sigma: ${sigma}"
            
            python /home/iwatts/catastrophic-forgetting/new_utils/evaluate_perturbed.py \
                --gcs_dir "${BASE_MODEL_PATH}" \
                --model_name "${PERTURBED_MODEL_NAME}" \
                --output_gcs_dir "gs://cmu-gpucloud-iwatts/outputs/${PROJECT}/ModelEvaluation" \
                --dtype "uint32"
            
            echo "✅ Completed evaluation for ${PERTURBED_MODEL_NAME}"
            echo ""
        done
    done
done