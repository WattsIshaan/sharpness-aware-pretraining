#!/usr/bin/env python3
"""
Download midtrained checkpoints for CPT experiments to a fixed cache directory.
Run this once before launching CPT jobs to avoid re-downloading on every run.

Checkpoints: 5B midtrain tokens, adamw + sam (rho 5e-2, 1e-1, 1.5e-1, 2e-1)
"""

import argparse
import os
import subprocess
import sys

# Add project root for imports; init Project before importing launch (which reads config)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_checkpoints_to_download(MidtrainedModel):
    """Return MidtrainedModel instances for the 5 checkpoints to cache."""
    return [
        MidtrainedModel(
            optimizer="adamw",
            midtrain_tokens=50,
            global_train_batch_size=1024,
        ),
        MidtrainedModel(
            optimizer="sam",
            sam_rho=5e-2,
            midtrain_tokens=50,
            global_train_batch_size=1024,
        ),
        # MidtrainedModel(
        #     optimizer="sam",
        #     sam_rho=1e-1,
        #     midtrain_tokens=5,
        #     global_train_batch_size=1024,
        # ),
        # MidtrainedModel(
        #     optimizer="sam",
        #     sam_rho=1.5e-1,
        #     midtrain_tokens=5,
        #     global_train_batch_size=1024,
        # ),
        # MidtrainedModel(
        #     optimizer="sam",
        #     sam_rho=2e-1,
        #     midtrain_tokens=5,
        #     global_train_batch_size=1024,
        # ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Download CPT midtrained checkpoints to cache")
    parser.add_argument(
        "--project",
        type=str,
        default="1b-experiments",
        help="Project name (default: 1b-experiments)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    from experiments import Project  # type: ignore
    Project.init(args.project)

    from launch import globals as G
    from launch.artifacts import MidtrainedModel
    gs_root = G.GS_PATH
    cache_dir = G.get_cpt_checkpoint_cache_dir()

    print(f"Cache directory: {cache_dir}")
    print(f"GS root: {gs_root}")
    print()

    checkpoints = get_checkpoints_to_download(MidtrainedModel)
    for model in checkpoints:
        pre_ckpt_rel = model.checkpoint_relpath
        gs_path = os.path.join(gs_root, pre_ckpt_rel)
        local_path = os.path.join(cache_dir, pre_ckpt_rel)

        print(f"Downloading: {model.run_name}")
        print(f"  From: {gs_path}")
        print(f"  To:   {local_path}")

        if args.dry_run:
            print(f"  [DRY-RUN] Would run: gsutil -m rsync -r -c {gs_path}/ {local_path}/")
            continue

        os.makedirs(local_path, exist_ok=True)
        cmd = ["gsutil", "-m", "rsync", "-r", "-c", f"{gs_path}/", f"{local_path}/"]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            sys.exit(1)
        print(f"  Done")
        print()

    print("All checkpoints downloaded successfully.")


if __name__ == "__main__":
    main()
