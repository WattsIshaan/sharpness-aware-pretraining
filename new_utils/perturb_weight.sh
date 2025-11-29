#!/usr/bin/env bash

#SBATCH --job-name=perturb_weight
#SBATCH --output=logs/perturb_weight_%j.out
#SBATCH --error=logs/perturb_weight_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 
#SBATCH --partition=general
#SBATCH --requeue

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv313


# List of sigma values
SIGMAS=(0.009 0.0095 0.013 0.015 0.017 0.022 0.025 0.02 0.01)

# List of model paths
MODELS=(
"gs://cmu-gpucloud-catheri4/outputs/adamw/PretrainedModel/OLMo-tk4B-adamw-lr3e-4-wd1e-1-bs256/final-unsharded/model.pt"
"gs://cmu-gpucloud-catheri4/outputs/adamw/PretrainedModel/OLMo-tk8B-adamw-lr3e-4-wd1e-1-bs256/final-unsharded/model.pt"
"gs://cmu-gpucloud-catheri4/outputs/adamw/PretrainedModel/OLMo-tk16B-adamw-lr3e-4-wd1e-1-bs256/final-unsharded/model.pt"
"gs://cmu-gpucloud-catheri4/outputs/adamw/PretrainedModel/OLMo-tk32B-adamw-lr3e-4-wd1e-1-bs256/final-unsharded/model.pt"
"gs://cmu-gpucloud-catheri4/outputs/adamw/PretrainedModel/OLMo-tk64B-adamw-lr3e-4-wd1e-1-bs256/final-unsharded/model.pt"
)

# Loop over sigmas and model paths

for sigma in "${SIGMAS[@]}"; do
    echo "Running with sigma=${sigma}"
    for model in "${MODELS[@]}"; do
        echo "  Model: ${model}"
        python /home/catheri4/catastrophic-forgetting/new_utils/perturb_weights.py \
            --gcs_model_path "${model}" \
            --sigma "${sigma}"
    done
done

echo "All runs completed."

# Run the downloader

