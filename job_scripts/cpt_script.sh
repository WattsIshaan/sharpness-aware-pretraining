#!/usr/bin/env bash

#SBATCH --job-name=olmo-20M-CPT
#SBATCH --output=logs/cpt/%j.out
#SBATCH --error=logs/cpt/%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

# Example usage:
# sbatch job_scripts/cpt_script.sh lionw

if [ $# -ne 1 ]; then
  echo "Usage: $0 <optimizer>"
  echo "Example: $0 lionw"
  exit 1
fi

OPTIMIZER="$1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate olmo

for val in 4 8 16 32 64; do
  for lr in 1e-3 2e-4 4e-5; do  
    CONFIG="new_configs/cpt/${OPTIMIZER}/lr_${lr}/OLMo-20M-${val}B-CPT.yaml"
    if [ ! -f "$CONFIG" ]; then
      echo "Config file $CONFIG does not exist!"
      continue
    fi
    echo "Running OLMo-20M-${val}B-CPT with lr ${lr} and optimizer ${OPTIMIZER}"
    torchrun --master_port=29501 --nproc_per_node=4 scripts/train.py "$CONFIG" --save_overwrite
  done
done