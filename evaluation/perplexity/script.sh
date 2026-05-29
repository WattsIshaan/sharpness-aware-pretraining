#!/bin/bash
#SBATCH --job-name=alpaca_split_perplexity
#SBATCH --output=/outputs/pplx/%j.out
#SBATCH --error=/outputs/pplx/%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G


# Specfic to user environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv310

# Run split
python /home/{ANDREW_ID}/catastrophic-forgetting/evaluation/perplexity/evaluate_perplexity.py \
 --dataset_path {DATA PATH} \
 --model_name {MODEL PATH}
