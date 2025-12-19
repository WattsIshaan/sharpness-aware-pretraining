#!/usr/bin/env bash

#SBATCH --job-name=tokenize_data
#SBATCH --output=logs/tokenize_data_%j.out
#SBATCH --error=logs/tokenize_data_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 
#SBATCH --partition=general
#SBATCH --requeue

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv313

# Set paths
SCRIPT_DIR="/home/catheri4/catastrophic-forgetting/new_utils"
TOKENIZER_PATH="/home/catheri4/catastrophic-forgetting/olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
OUTPUT_BASE_DIR="/tmp/tokenized_data"
GCS_BUCKET="gs://cmu-gpucloud-catheri4/datasets"  # Update with your bucket

# Tokenization parameters
MAX_TOKENS=100000000  # 100M tokens 
SEQ_LEN=1024          # Sequence length
BUFFER_MULT=1.4       # Buffer multiplier for early stopping

# Create logs directory if it doesn't exist
mkdir -p logs

# If a dataset is provided as argument, use it. Otherwise process all datasets
if [ -n "$1" ]; then
    DATASETS=("$@")
else
    # Default datasets to process
    # Format: "dataset_name" or "dataset_name:config_name"
    declare -a DATASETS=(
        "allenai/social_i_qa"
        #"openai/gsm8k:main"
         # Requires config (main or socratic)
        # Other options:
        # "bigcode/starcoderdata"  # For code data
        # "HuggingFaceH4/ultrachat_200k"  # Chat data
        # "HuggingFaceH4/ultrafeedback_binarized"  # Preference data
    )
fi

# Process each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "================================================"
    echo "Processing dataset: $DATASET"
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "================================================"

    # Create output directory for this dataset
    # Replace / and : with _ for directory name
    DATASET_NAME=$(echo $DATASET | sed 's/\//_/g' | sed 's/:/_/g')
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${DATASET_NAME}"
    mkdir -p $OUTPUT_DIR

    # Run tokenization
    cd $SCRIPT_DIR
    python tokenize_data.py \
        $OUTPUT_DIR \
        -d $DATASET \
        -t $TOKENIZER_PATH \
        -s $SEQ_LEN \
        -m $MAX_TOKENS \
        -b $BUFFER_MULT \
        -j 8 \
        -g $GCS_BUCKET

    echo "================================================"
    echo "Completed processing: $DATASET"
    echo "================================================"
done
