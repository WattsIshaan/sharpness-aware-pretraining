#!/usr/bin/env python3
"""
Script to evaluate perturbed model checkpoints on specified datasets.

Downloads a perturbed checkpoint from GCS, downloads evaluation data files,
evaluates the model on each dataset, and uploads results to GCS.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def download_from_gcs(gcs_path: str, local_path: str, directory: bool = False) -> None:
    """Download a file or directory from GCS to local path."""
    if directory:
        # Ensure local directory exists
        os.makedirs(local_path, exist_ok=True)
        subprocess.check_call(["gsutil", "-m", "rsync", "-r", gcs_path + "/", local_path + "/"])
    else:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        subprocess.check_call(["gsutil", "-m", "cp", gcs_path, local_path])


def upload_to_gcs(local_path: str, gcs_path: str, directory: bool = False) -> None:
    """Upload a local file or directory to GCS."""
    if directory:
        subprocess.check_call(["gsutil", "-m", "rsync", "-r", local_path + "/", gcs_path + "/"])
    else:
        # Ensure parent directory exists in GCS (gsutil will create it)
        subprocess.check_call(["gsutil", "-m", "cp", local_path, gcs_path])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate perturbed model checkpoint on specified datasets"
    )
    parser.add_argument(
        "--gcs_dir",
        type=str,
        default="gs://cmu-gpucloud-iwatts/outputs/large-scale-experiments/PerturbedModel",
        help="GCS directory path (e.g., gs://bucket/path/to/models)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (directory name containing final-unsharded/model.pt)"
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        default="gs://cmu-gpucloud-jspringe/shared/datasets/OLMo/dclm/val/preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer_part-187-00004-2M.npy",
        help="Comma-separated GCS paths to evaluation data files (numpy uint16 format)"
    )
    parser.add_argument(
        "--output_gcs_dir",
        type=str,
        default="gs://cmu-gpucloud-iwatts/outputs/large-scale-experiments/ModelEvaluation",
        help="GCS directory path for storing results (default: gs://cmu-gpucloud-iwatts/outputs/large-scale-experiments/ModelEvaluation)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Size of chunks to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation, e.g., 'cuda', 'cpu', or 'cuda:0'"
    )
    
    args = parser.parse_args()
    
    # Create temporary working directory
    work_dir = tempfile.mkdtemp(prefix="eval_perturbed_")
    cleanup_work_dir = True
    
    try:        
        # Step 1: Download checkpoint from GCS
        # Expected path: gcs_dir/model_name/final-unsharded/model.pt
        gcs_ckpt_dir = f"{args.gcs_dir.rstrip('/')}/{args.model_name}/final-unsharded"
        
        log.info(f"📥 Downloading checkpoint from {gcs_ckpt_dir}...")
        
        # Download to a temporary checkpoint directory
        local_ckpt_dir = os.path.join(work_dir, "checkpoint")
        os.makedirs(local_ckpt_dir, exist_ok=True)
        download_from_gcs(gcs_ckpt_dir, local_ckpt_dir, directory=True)
        
        # Verify model.pt exists
        model_pt_path = os.path.join(local_ckpt_dir, "model.pt")
        if not os.path.exists(model_pt_path):
            raise FileNotFoundError(f"Could not find model.pt in downloaded checkpoint at {local_ckpt_dir}")
        
        log.info(f"✅ Checkpoint downloaded to {local_ckpt_dir}")
        
        # Step 2: Download evaluation data files from GCS
        data_gcs_paths = [p.strip() for p in args.data_paths.split(",") if p.strip()]
        local_data_paths = []
        
        log.info(f"📥 Downloading {len(data_gcs_paths)} evaluation data file(s)...")
        local_data_dir = os.path.join(work_dir, "data")
        os.makedirs(local_data_dir, exist_ok=True)
        
        for i, gcs_data_path in enumerate(data_gcs_paths):
            # Extract filename from GCS path
            filename = os.path.basename(gcs_data_path.rstrip("/"))
            local_data_path = os.path.join(local_data_dir, filename)
            
            log.info(f"  Downloading {gcs_data_path} -> {local_data_path}")
            download_from_gcs(gcs_data_path, local_data_path, directory=False)
            local_data_paths.append(local_data_path)
        
        log.info(f"✅ All data files downloaded")
        
        # Step 3: Prepare output path
        # Output: output_gcs_dir/model_name.json
        output_gcs_path = f"{args.output_gcs_dir.rstrip('/')}/{args.model_name}-eval.json"
        local_output_path = os.path.join(work_dir, "results.json")
        
        # Step 4: Run evaluation using evaluate.py script
        log.info(f"🔍 Starting evaluation on {len(local_data_paths)} dataset(s)...")
        
        # Build comma-separated list for evaluator
        data_path_arg = ','.join(local_data_paths)
        
        # Path to evaluation script
        eval_script = os.path.join('scripts', 'evaluate.py')
        
        # Run evaluation
        log.info(f"Running evaluation script: {eval_script}")
        subprocess.check_call([
            'python', eval_script,
            f"--model_path", local_ckpt_dir,
            f"--device", args.device,
            f"--data_path", data_path_arg,
            f"--chunk_size", str(args.chunk_size),
            f"--output_path", local_output_path,
        ])
        
        log.info(f"✅ Evaluation complete")
        
        # Step 6: Upload results to GCS
        log.info(f"☁️ Uploading results to {output_gcs_path}...")
        upload_to_gcs(local_output_path, output_gcs_path, directory=False)
        log.info(f"✅ Results uploaded successfully")
        
        log.info(f"🎉 Done! Evaluation results saved to {output_gcs_path}")
        return 0
    
    except Exception as e:
        log.error(f"❌ Error: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup temporary directory if we created it
        if cleanup_work_dir and os.path.exists(work_dir):
            import shutil
            log.info(f"🧹 Cleaning up temporary directory {work_dir}...")
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    sys.exit(main())

