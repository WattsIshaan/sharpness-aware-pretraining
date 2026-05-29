#!/usr/bin/env bash

#SBATCH --job-name=OLMo-20M-multirun
#SBATCH --output=logs/pretrain/%j.out
#SBATCH --error=logs/pretrain/%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Sample usage:
# sbatch job_scripts/pretrain_script.sh adamw
# sbatch job_scripts/pretrain_script.sh lionw

if [ $# -ne 1 ]; then
    echo "Usage: $0 <adamw|lionw>"
    exit 1
fi

OPTIMIZER="$1"
if [[ "$OPTIMIZER" != "adamw" && "$OPTIMIZER" != "lionw" ]]; then
    echo "Optimizer must be 'adamw' or 'lionw'"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate olmo

for TOKENS in 4 8 16 32; do
    CONFIG="new_configs/pretrain/${OPTIMIZER}/OLMo-20M-${TOKENS}B-PreTrain.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "Config file $CONFIG does not exist!"
        exit 1
    fi
    echo "Running with config $CONFIG"
    torchrun --nproc_per_node=8 scripts/train.py "$CONFIG" --save_overwrite
done