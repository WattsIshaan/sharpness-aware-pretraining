#!/bin/bash
#SBATCH --job-name=i
#SBATCH --partition=general
#SBATCH --output=logs/debug_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=256GB
#SBATCH --time=2-00:00:00
#SBATCH --requeue

while true; do
  sleep 1h
done