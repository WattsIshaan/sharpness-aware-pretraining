#!/usr/bin/env bash

#SBATCH --job-name=tokenize_data
#SBATCH --output=logs/tokenize_data_%j.out
#SBATCH --error=logs/tokenize_data_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1 
#SBATCH --partition=general

source ~/miniconda3/etc/profile.d/conda.sh
conda activate forgetting

# Set paths
SCRIPT_DIR="/home/iwatts/catastrophic-forgetting/new_utils"
TOKENIZER_PATH="/home/iwatts/catastrophic-forgetting/olmo_data/tokenizers/allenai_dolma2.json"
OUTPUT_BASE_DIR="/tmp/iwatts/tokenized_data"
GCS_BUCKET="gs://cmu-gpucloud-jspringe/shared/datasets/OLMo"

# --- Tokenization Parameters ---
MAX_TOKENS=100000000 
SEQ_LEN=1024

# Special Tokens (Adjust these if using a non-Dolma tokenizer)
EOS_TOKEN_ID=100257
PAD_TOKEN_ID=100277
# -------------------------------

mkdir -p logs

if [ -n "$1" ]; then
    DATASETS=("$@")
else
    declare -a DATASETS=("m-a-p/MusicPile-sft") #"bigcode/starcoderdata" "allenai/tulu-3-sft-mixture")
fi

for DATASET in "${DATASETS[@]}"; do
    DATASET_DIR_NAME=$(echo $DATASET | sed 's/\//_/g' | sed 's/:/_/g')
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${DATASET_DIR_NAME}"
    mkdir -p $OUTPUT_DIR

    echo "Starting processing for: $DATASET"
    
    cd $SCRIPT_DIR
    python tokenize_data.py \
        $OUTPUT_DIR \
        -d $DATASET \
        -t $TOKENIZER_PATH \
        -s $SEQ_LEN \
        -m $MAX_TOKENS \
        --eos $EOS_TOKEN_ID \
        --pad $PAD_TOKEN_ID \
        -j 8 \
        -g $GCS_BUCKET
done