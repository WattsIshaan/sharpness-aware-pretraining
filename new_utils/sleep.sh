#!/bin/bash
#SBATCH --job-name=i
#SBATCH --partition=flame-earlybirds
#SBATCH --qos=earlybird_qos
#SBATCH --output=logs/debug_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H100:8
#SBATCH --mem=2000000M
#SBATCH --time=2-00:00:00
#SBATCH --requeue

while true; do
  sleep 1h
done