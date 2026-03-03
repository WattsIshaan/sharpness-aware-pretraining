#!/bin/bash
#SBATCH --job-name=midtrain_data_%j
#SBATCH --output=logs/midtrain_data_%j.out
#SBATCH --error=logs/midtrain_data_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

json_path="/home/iwatts/catastrophic-forgetting/launch/utils/midtrain_data.json"
dest_bucket="gs://cmu-gpucloud-jspringe/shared/datasets/OLMo/midtrain/train"
tmp_dir="$(mktemp -d)"
prefix="http://olmo-data.org/preprocessed/"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but not found in PATH." >&2
  exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil is required but not found in PATH." >&2
  exit 1
fi

while IFS= read -r url; do
  if [[ "${url}" != "${prefix}"* ]]; then
    echo "Skipping unexpected URL (missing prefix): ${url}" >&2
    continue
  fi

  rel="${url#${prefix}}"
  filename="${rel//\//_}"
  local_path="${tmp_dir}/${filename}"

  echo "Downloading ${url}"
  curl -L -o "${local_path}" "${url}"

  echo "Uploading ${local_path} -> ${dest_bucket}/${filename}"
  gsutil cp "${local_path}" "${dest_bucket}/${filename}"
done < <(jq -r '.train_data_paths[]' "${json_path}")
