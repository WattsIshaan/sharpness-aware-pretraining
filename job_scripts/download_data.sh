#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --output=/home/catheri4/catastrophic-forgetting/logs/download_data/%j.out
#SBATCH --error=/home/catheri4/catastrophic-forgetting/logs/download_data/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=preempt
#SBATCH --requeue

# Set up environment
cd /home/catheri4/catastrophic-forgetting

# Create logs directory if it doesn't exist
mkdir -p logs

# Default parameters (can be overridden by command line arguments)
CONFIG_FILE="${1:-configs/tiny/OLMo-20M-starcoder.yaml}"
OUTPUT_DIR="${2:-/project/flame/catheri4/datasets/starcoder/}"
DATA_TYPE="${3:-all}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env310

# Example usage to override one argument:
# sbatch job_scripts/download_data.sh configs/tiny/OLMo-20M.yaml /project/flame/iwatts/datasets/c4/ val
# Or, to override just the data type (third argument), keeping defaults for the first two:
# sbatch job_scripts/download_data.sh "" "" val

# Print parameters
echo "Configuration file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Data type: $DATA_TYPE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Run the download script
echo "Starting data download..."
python new_utils/download_data.py "$CONFIG_FILE" "$OUTPUT_DIR" "$DATA_TYPE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Download completed successfully!"
    echo "Files downloaded to: $OUTPUT_DIR"
    echo "Files in output directory:"
    ls -la "$OUTPUT_DIR"
else
    echo "Download failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
echo "Job completed!"
