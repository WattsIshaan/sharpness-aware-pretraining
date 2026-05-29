#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --array=0-1 

#modify array wrt to how many separate elems in task_groups


TASK_GROUPS=("lambada,gsm8k" "winogrande,mmlu") #define however

TASKS=${TASK_GROUPS[$SLURM_ARRAY_TASK_ID]}

echo "Running tasks: $TASKS"

#specific to personal setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv310 
cd ~/utils/eval/lm-evaluation-harness

pip install -e .

# Run evaluation
lm_eval \
    --model hf \
    --model_args "pretrained={MODEL_PATH}" \
    --tasks $TASKS \
    --device cuda:0 \
    --batch_size 4 \
    --output_path #path to results