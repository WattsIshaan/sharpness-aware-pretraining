"""Global constants and paths for the Wikipedia pretraining experiment."""

import subprocess
import os
import random
import string

# Remote existence checking
CHECK_EXISTS_REMOTE = True  # Set to False to skip remote existence checks for speed

WANDB_ENTITY = 'iwatts-carnegie-mellon-university'
PROJECT_NAME = 'sam-ablation'
LOCAL_DATA_PATH = f'/tmp/iwatts/outputs/{PROJECT_NAME}'
LOCAL_HF_PATH = f'/tmp/iwatts/hf'
CODE_PATH = ''
OLMO_PATH = '/home/iwatts/catastrophic-forgetting'
GS_PATH = f'gs://cmu-gpucloud-iwatts/outputs/{PROJECT_NAME}'
GS_DATA_PATH = f'gs://cmu-gpucloud-iwatts'


# Cache for remote GS file list
_remote_files_cache = None


def get_remote_files():
    """Get list of all remote files in GS bucket. Cached for performance."""
    global _remote_files_cache
    
    if not CHECK_EXISTS_REMOTE:
        return set()
    
    if _remote_files_cache is None:
        try:
            result = subprocess.run(
                ['gsutil', 'ls', '-r', GS_PATH],
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

def get_random_local_path():
    return os.path.join(LOCAL_DATA_PATH, ''.join(random.choices(string.ascii_letters, k=8)))
