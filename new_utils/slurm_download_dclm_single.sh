#!/usr/bin/env bash

#SBATCH --job-name=download_dclm
#SBATCH --output=logs/download_dclm_%j.out
#SBATCH --error=logs/download_dclm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate olmo

# Run the downloader
wget -O datasets/dclm/val/preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer_part-187-00004.npy http://olmo-data.org/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-187-00004.npy
