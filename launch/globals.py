# PROJECT_NAME=sam-ablation; mkdir -p ~/.experiments/project-config && python3 -c 'import json,os; P=os.environ.get("PROJECT_NAME","sam-ablation"); d={"CHECK_EXISTS_REMOTE": True, "WANDB_ENTITY":"iwatts-carnegie-mellon-university", "PROJECT_NAME": P, "LOCAL_DATA_PATH": f"/tmp/iwatts/outputs/{P}", "LOCAL_HF_PATH": "/tmp/iwatts/hf", "CODE_PATH": "", "OLMO_PATH": "/home/iwatts/catastrophic-forgetting", "GS_PATH": f"gs://cmu-gpucloud-iwatts/outputs/{P}", "GS_DATA_PATH": "gs://cmu-gpucloud-iwatts"}; print(json.dumps(d, indent=2))' > ~/.experiments/project-config/$PROJECT_NAME.json
"""Global constants and paths for the Wikipedia pretraining experiment."""

import subprocess
import os
import random
import string
import json
from typing import Optional, Set, cast
from experiments import Project


# Cache for remote GS file list
_remote_files_cache = None




def get_remote_files() -> Set[str]:
    """Get list of all remote files in GS bucket. Cached for performance."""
    global _remote_files_cache
    
    if Project.config.CHECK_EXISTS_REMOTE is None:
        raise RuntimeError('Globals not initialized. Call load_project(project_name) first.')

    if not Project.config.CHECK_EXISTS_REMOTE:
        return set()
    
    if _remote_files_cache is None:
        try:
            if Project.config.GS_PATH is None:
                raise RuntimeError('Globals not initialized. Call load_project(project_name) first.')
            gs_path: str = cast(str, Project.config.GS_PATH)
            result = subprocess.run(
                ['gsutil', 'ls', '-r', gs_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                # Parse output to get all file paths
                _remote_files_cache = set(
                    line.strip() 
                    for line in result.stdout.split('\n') 
                    if line.strip() and not line.strip().endswith(':')
                )
            else:
                # If command fails, return empty set
                _remote_files_cache = set()
        except Exception:
            # If any error occurs, return empty set
            _remote_files_cache = set()
    
    return _remote_files_cache

def get_random_local_path() -> str:
    if Project.config.LOCAL_DATA_PATH is None:
        raise RuntimeError('Globals not initialized. Call load_project(project_name) first.')
    local_data_path: str = cast(str, Project.config.LOCAL_DATA_PATH)
    return os.path.join(local_data_path, ''.join(random.choices(string.ascii_letters, k=8)))
