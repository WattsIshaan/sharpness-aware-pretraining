#!/usr/bin/env bash

#SBATCH --job-name=weight_diff_all
#SBATCH --output=logs/weight_diff_all_%j.out
#SBATCH --error=logs/weight_diff_all_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 
#SBATCH --partition=general

# Exit on error
set -e

# Print job info
echo "=========================================="
echo "Weight Difference Computation - All Datasets"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting time: $(date)"
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv313
cd catastrophic-forgetting

# Create output directories
mkdir -p results
mkdir -p logs

# Configuration
GS_PATH="gs://cmu-gpucloud-catheri4/outputs/muon"
#BASE_MODELS="OLMo2-20m-tk4B-adamw-lr2e-3-wd1e-1-bs512 OLMo2-20m-tk8B-adamw-lr1e-3-wd1e-1-bs512 OLMo2-20m-tk16B-adamw-lr1e-3-wd1e-1-bs512 OLMo2-20m-tk32B-adamw-lr5e-4-wd1e-1-bs512 OLMo2-20m-tk64B-adamw-lr3e-4-wd1e-1-bs512"
BASE_MODELS="OLMo2-20m-tk4B-muon-lr2e-2-wd1e-1-bs512-muon_lr8e-3 OLMo2-20m-tk8B-muon-lr2e-2-wd1e-1-bs512-muon_lr4e-3 OLMo2-20m-tk16B-muon-lr2e-2-wd1e-1-bs512-muon_lr4e-3 OLMo2-20m-tk32B-muon-lr2e-2-wd1e-1-bs512-muon_lr4e-3 OLMo2-20m-tk64B-muon-lr2e-2-wd1e-1-bs512-muon_lr4e-3"
DATASETS=("starcoder" "alpaca" "tulu")
OPTIMIZER=""  # Optional: filter by optimizer (e.g., "muon")
LEARNING_RATE=""  # Optional: filter by LR (e.g., "1.00e-04")

echo "Configuration:"
echo "  GCS Path: $GS_PATH"
echo "  Datasets: ${DATASETS[@]}"
echo "  Base Models: $BASE_MODELS"
echo "  Optimizer filter: ${OPTIMIZER:-none}"
echo "  Learning rate filter: ${LEARNING_RATE:-none}"
echo ""

# Process each dataset sequentially
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing dataset: $dataset"
    echo "=========================================="
    echo "Started at: $(date)"
    
    output_csv="results/weight_diff_muon_20M_${dataset}_$(date +%Y%m%d).csv"
    
    # Build command
    CMD="python new_utils/compute_weight_differences.py \
        --cpt-dataset $dataset \
        --base-models $BASE_MODELS \
        --gs-path $GS_PATH \
        --output-csv $output_csv"
    
    # Add optional filters
    if [ -n "$OPTIMIZER" ]; then
        CMD="$CMD --optimizer $OPTIMIZER"
    fi
    
    if [ -n "$LEARNING_RATE" ]; then
        CMD="$CMD --learning-rate $LEARNING_RATE"
    fi
    
    echo "Running command:"
    echo "$CMD"
    echo ""
    
    # Run the script
    $CMD
    
    echo ""
    echo "Finished dataset: $dataset at $(date)"
    echo "Results saved to: $output_csv"
    echo ""
done

echo "=========================================="
echo "All datasets processed!"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Output files:"
for dataset in "${DATASETS[@]}"; do
    output_csv="results/weight_diff_${dataset}_$(date +%Y%m%d).csv"
    if [ -f "$output_csv" ]; then
        echo "  ✓ $output_csv"
    else
        echo "  ✗ $output_csv (not found)"
    fi
done
