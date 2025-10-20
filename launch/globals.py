# PROJECT_NAME=sam-ablation; mkdir -p ~/.experiments/project-config && python3 -c 'import json,os; P=os.environ.get("PROJECT_NAME","sam-ablation"); d={"CHECK_EXISTS_REMOTE": True, "WANDB_ENTITY":"iwatts-carnegie-mellon-university", "PROJECT_NAME": P, "LOCAL_DATA_PATH": f"/tmp/iwatts/outputs/{P}", "LOCAL_HF_PATH": "/tmp/iwatts/hf", "CODE_PATH": "", "OLMO_PATH": "/home/iwatts/catastrophic-forgetting", "GS_PATH": f"gs://cmu-gpucloud-iwatts/outputs/{P}", "GS_DATA_PATH": "gs://cmu-gpucloud-iwatts"}; print(json.dumps(d, indent=2))' > ~/.experiments/project-config/$PROJECT_NAME.json
"""Global constants and paths for the Wikipedia pretraining experiment."""

import subprocess
import os
import random
import string
import json
from typing import Optional, Set, cast

# Module variables are declared but uninitialized until load_project is called.
CHECK_EXISTS_REMOTE: Optional[bool] = None
WANDB_ENTITY: Optional[str] = None
PROJECT_NAME: Optional[str] = None
LOCAL_DATA_PATH: Optional[str] = None
LOCAL_HF_PATH: Optional[str] = None
CODE_PATH: Optional[str] = None
OLMO_PATH: Optional[str] = None
GS_PATH: Optional[str] = None
GS_DATA_PATH: Optional[str] = None


# Cache for remote GS file list
_remote_files_cache = None


def load_project(project_name: str) -> None:
    """Load project configuration strictly from ~/.experiments/project-config/{project_name}.json.
    Raises FileNotFoundError if the config does not exist, or ValueError if required keys are missing.
    """
    global CHECK_EXISTS_REMOTE, WANDB_ENTITY, PROJECT_NAME, LOCAL_DATA_PATH, LOCAL_HF_PATH
    global CODE_PATH, OLMO_PATH, GS_PATH, GS_DATA_PATH, _remote_files_cache

    config_path = os.path.expanduser(f'~/.experiments/project-config/{project_name}.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Project config not found: {config_path}')

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    required_keys = [
        'CHECK_EXISTS_REMOTE', 'WANDB_ENTITY', 'PROJECT_NAME', 'LOCAL_DATA_PATH',
        'LOCAL_HF_PATH', 'CODE_PATH', 'OLMO_PATH', 'GS_PATH', 'GS_DATA_PATH'
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f'Missing required config keys in {config_path}: {missing}')

    CHECK_EXISTS_REMOTE = bool(cfg['CHECK_EXISTS_REMOTE'])
    WANDB_ENTITY = cfg['WANDB_ENTITY']
    PROJECT_NAME = cfg['PROJECT_NAME']
    LOCAL_DATA_PATH = cfg['LOCAL_DATA_PATH']
    LOCAL_HF_PATH = cfg['LOCAL_HF_PATH']
    CODE_PATH = cfg['CODE_PATH']
    OLMO_PATH = cfg['OLMO_PATH']
    GS_PATH = cfg['GS_PATH']
    GS_DATA_PATH = cfg['GS_DATA_PATH']

    # Reset remote files cache on project switch
    _remote_files_cache = None


def get_remote_files() -> Set[str]:
    """Get list of all remote files in GS bucket. Cached for performance."""
    global _remote_files_cache
    
    if CHECK_EXISTS_REMOTE is None:
        raise RuntimeError('Globals not initialized. Call load_project(project_name) first.')

    if not CHECK_EXISTS_REMOTE:
        return set()
    
    if _remote_files_cache is None:
        try:
            if GS_PATH is None:
                raise RuntimeError('Globals not initialized. Call load_project(project_name) first.')
            gs_path: str = cast(str, GS_PATH)
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
    if LOCAL_DATA_PATH is None:
        raise RuntimeError('Globals not initialized. Call load_project(project_name) first.')
    local_data_path: str = cast(str, LOCAL_DATA_PATH)
    return os.path.join(local_data_path, ''.join(random.choices(string.ascii_letters, k=8)))
