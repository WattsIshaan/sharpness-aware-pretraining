#!/usr/bin/env bash

#SBATCH --job-name=download_dclm
#SBATCH --output=logs/download_dclm_%j.out
#SBATCH --error=logs/download_dclm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --requeue

source ~/miniconda3/etc/profile.d/conda.sh
conda activate olmo

# Run the downloader
python new_utils/download_dclm.py
