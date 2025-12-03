#!/usr/bin/env python3
"""
Script to perturb model weights by adding Gaussian noise and save to GCS.

Takes a model from a GCS bucket, perturbs all parameters by epsilon ~ N(0, sigma^2),
and saves the perturbed model back to GCS.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """Download a file from GCS to local path."""
    subprocess.check_call(["gsutil", "-m", "cp", gcs_path, local_path])


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload a local file to GCS."""
    subprocess.check_call(["gsutil", "-m", "cp", local_path, gcs_path])


def rsync_from_gcs(gcs_source: str, local_dest: str) -> None:
    """Rsync a directory from GCS to local path."""
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", gcs_source, local_dest])


def rsync_to_gcs(local_source: str, gcs_dest: str) -> None:
    """Rsync a directory from local to GCS."""
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", local_source, gcs_dest])


def copy_gcs_directory(gcs_source: str, gcs_dest: str) -> None:
    """Copy a directory from one GCS location to another."""
    # Use rsync to copy directory contents
    # First ensure destination exists by creating a dummy file, then remove it
    # Actually, rsync will create the directory if needed
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", gcs_source + "/", gcs_dest + "/"])

def perturb_state_dict(state_dict: dict, gamma: float, seed: Optional[int] = None) -> dict:
    """
    Perturb parameters in a state_dict by adding Gaussian noise scaled by
    the initialization norm: std = gamma * ||W0||_F.

    Args:
        state_dict: Current model state_dict
        init_state_dict: Initialization state_dict (same keys)
        gamma: Scalar multiplier for perturbation scale
        seed: Optional random seed
    
    Returns:
        Perturbed state dictionary
    """
    if seed is not None:
        torch.manual_seed(seed)

    perturbed_state = {}

    for name, W in state_dict.items():
        # Only perturb float parameters
        if W.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            perturbed_state[name] = W
            continue

        # Frobenius norm of weight
        s = torch.norm(W)     # ||W||_F

        # std = gamma / s
        std = gamma / s

        noise = torch.randn_like(W) * std
        perturbed_state[name] = W + noise

    return perturbed_state



def load_model_from_gcs(gcs_path: str, device: str = "cpu") -> dict:
    """
    Load a PyTorch model state_dict from GCS.
    
    Args:
        gcs_path: GCS path to the model file (e.g., gs://bucket/path/model.pt)
        device: Device to load the model on
    
    Returns:
        Model state dictionary
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        print(f"📥 Downloading model from {gcs_path}...")
        download_from_gcs(gcs_path, tmp_path)
        
        print(f"📂 Loading model state dict...")
        state_dict = torch.load(tmp_path, map_location=device)
        
        # Handle different formats: state_dict might be wrapped in a dict
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        
        return state_dict
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_model_to_gcs(state_dict: dict, gcs_path: str) -> None:
    """
    Save a PyTorch model state_dict to GCS.
    
    Args:
        state_dict: Model state dictionary
        gcs_path: GCS path to save the model (e.g., gs://bucket/path/model_perturbed.pt)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        print(f"💾 Saving perturbed model locally...")
        torch.save(state_dict, tmp_path)
        
        print(f"☁️ Uploading perturbed model to {gcs_path}...")
        upload_to_gcs(tmp_path, gcs_path)
        print(f"✅ Successfully uploaded perturbed model to {gcs_path}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_perturbed_model_name(model_name: str, sigma: float) -> str:
    """
    Generate perturbed model name with sigma suffix.
    
    Args:
        model_name: Original model name
        sigma: Standard deviation value to append to the directory name
    
    Returns:
        Perturbed model name (e.g., "model_name_perturbed_1_00e-02")
    """
    # Format sigma value for directory name (use scientific notation)
    sigma_str = f"{sigma:.2e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
    return f"{model_name}_perturbed_{sigma_str}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perturb model weights with Gaussian noise and save to GCS"
    )
    parser.add_argument(
        "--gcs_dir",
        type=str,
        required=True,
        help="GCS directory path (e.g., gs://bucket/path/to/models)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (directory name containing final-unsharded/model.pt)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Standard deviation of Gaussian noise (N(0, sigma^2))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=64,
        help="Random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--output_gcs_dir",
        type=str,
        required=True,
        help="GCS directory path for storing perturbed model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (default: cpu)",
    )
    
    args = parser.parse_args()
    
    try:
        # Construct paths following evaluate_perturbed.py structure
        # Input: gcs_dir/model_name/final-unsharded/model.pt
        original_checkpoint_dir = f"{args.gcs_dir.rstrip('/')}/{args.model_name}"
        original_model_path = f"{original_checkpoint_dir}/final-unsharded/model.pt"
        
        # Output: output_gcs_dir/model_name_perturbed_{sigma}/final-unsharded/model.pt
        perturbed_model_name = get_perturbed_model_name(args.model_name, args.sigma)
        perturbed_checkpoint_dir = f"{args.output_gcs_dir.rstrip('/')}/{perturbed_model_name}"
        perturbed_model_path = f"{perturbed_checkpoint_dir}/final-unsharded/model.pt"
        
        print(f"📋 Original checkpoint directory: {original_checkpoint_dir}")
        print(f"📋 Original model path: {original_model_path}")
        print(f"📋 Perturbed checkpoint directory: {perturbed_checkpoint_dir}")
        print(f"📋 Perturbed model path: {perturbed_model_path}")
        
        # Step 1: Copy entire checkpoint directory to new location
        print(f"📦 Copying checkpoint directory from {original_checkpoint_dir} to {perturbed_checkpoint_dir}...")
        copy_gcs_directory(original_checkpoint_dir, perturbed_checkpoint_dir)
        print(f"✅ Checkpoint directory copied")
        
        # Step 2: Load and perturb the model
        print(f"📥 Loading model from {original_model_path}...")
        state_dict = load_model_from_gcs(original_model_path, device=args.device)
        
        print(f"🔧 Perturbing parameters with sigma={args.sigma}...")
        perturbed_state_dict = perturb_state_dict(state_dict, args.sigma, seed=args.seed)
        
        # Step 3: Replace model.pt in the perturbed checkpoint directory
        print(f"💾 Replacing model.pt in perturbed checkpoint...")
        save_model_to_gcs(perturbed_state_dict, perturbed_model_path)
        
        print("🎉 Done! Perturbed checkpoint created successfully.")
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

