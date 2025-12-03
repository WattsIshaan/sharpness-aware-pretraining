#!/usr/bin/env bash

#SBATCH --job-name=evaluate_perturbed
#SBATCH --output=logs/evaluate_perturbed_%j.out
#SBATCH --error=logs/evaluate_perturbed_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4 
#SBATCH --requeue
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu_qos
#SBATCH --account=aditirag

source ~/miniconda3/etc/profile.d/conda.sh
conda activate forgetting


# List of sigma values (must match perturb_weight.sh)
# SIGMAS=(0.009 0.0095 0.013 0.015 0.017 0.022 0.025 0.02 0.01)
SIGMAS=(0.03 0.04 0.05 0.075 0.1)
BASE_MODEL_PATH="gs://cmu-gpucloud-iwatts/outputs/large-scale-experiments/PerturbedModel"
OPTIM="adamw"

CKPT_STEPS=(40000 85000 165000 335000 670000) #60m
PT_TOKEN=400
MODEL_SIZE=60
# CKPT_STEPS=(15000 30000 55000 110000 220000 445000 890000) #20m
# PT_TOKEN=300
# MODEL_SIZE=20


# Function to format sigma for directory name (matches perturb_weights.py)
format_sigma() {
    local sigma=$1
    # Format as scientific notation and replace dots with underscores
    printf "%.2e" "$sigma" | sed 's/e-0/e-/g' | sed 's/e+0/e+/g' | sed 's/\./_/g'
}

# Loop over checkpoint steps and sigmas
for ckpt_step in "${CKPT_STEPS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        if [ "$OPTIM" == "sam_adamw" ]; then
            MODEL_NAME=OLMo-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr3e-4-wd1e-1-bs256-rho5e-2-anneal-ckpt${ckpt_step}
        else
            MODEL_NAME=OLMo-${MODEL_SIZE}m-tk${PT_TOKEN}B-${OPTIM}-lr3e-4-wd1e-1-bs256-anneal-ckpt${ckpt_step}
        fi
        SIGMA_STR=$(format_sigma "$sigma")
        PERTURBED_MODEL_NAME="${MODEL_NAME}_perturbed_${SIGMA_STR}"
        
        echo "Evaluating model: ${PERTURBED_MODEL_NAME}"
        echo "  Checkpoint step: ${ckpt_step}"
        echo "  Sigma: ${sigma}"
        
        python /home/iwatts/catastrophic-forgetting/new_utils/evaluate_perturbed.py \
            --gcs_dir "${BASE_MODEL_PATH}" \
            --model_name "${PERTURBED_MODEL_NAME}" \
            --output_gcs_dir "gs://cmu-gpucloud-iwatts/outputs/large-scale-experiments/ModelEvaluation"
        
        echo "✅ Completed evaluation for ${PERTURBED_MODEL_NAME}"
        echo ""
    done
done

