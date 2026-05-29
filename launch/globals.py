"""Global constants and paths for the Wikipedia pretraining experiment."""

import subprocess
import os
import random
import string
from typing import Optional, Set, cast
from experiments import Project # type: ignore

# Module variables are declared but uninitialized until load_project is called.
CHECK_EXISTS_REMOTE: Optional[bool] = Project.config.CHECK_EXISTS_REMOTE
DOWNLOAD_DATA: Optional[bool] = Project.config.DOWNLOAD_DATA
WANDB_ENTITY: Optional[str] = Project.config.WANDB_ENTITY
PROJECT_NAME: Optional[str] = Project.config.PROJECT_NAME
LOCAL_DATA_PATH: Optional[str] = Project.config.LOCAL_DATA_PATH
LOCAL_OUTPUT_PATH: Optional[str] = Project.config.LOCAL_OUTPUT_PATH
LOCAL_HF_PATH: Optional[str] = Project.config.LOCAL_HF_PATH
CODE_PATH: Optional[str] = Project.config.CODE_PATH
OLMO_PATH: Optional[str] = Project.config.OLMO_PATH
GS_PATH: Optional[str] = Project.config.GS_PATH
GS_DATA_PATH: Optional[str] = Project.config.GS_DATA_PATH
CLUSTER_NAME: Optional[str] = Project.config.CLUSTER_NAME


# Cache for remote GS file list - dict mapping folder paths to file sets
_remote_files_cache = {}


def get_remote_files(subfolder: Optional[str] = None) -> Set[str]:
    """Get list of all remote files in GS bucket. Cached for performance.
    
    Args:
        subfolder: Optional subfolder to list (e.g., 'PretrainedModel'). 
                   If None, lists entire bucket. If specified, only lists that folder.
    """
    global _remote_files_cache
    
    if Project.config.CHECK_EXISTS_REMOTE is None:
        raise RuntimeError('Globals not initialized. Call load_project(project_name) first.')

    if not Project.config.CHECK_EXISTS_REMOTE:
        return set()
    
    # Use subfolder as cache key, or 'ALL' for entire bucket
    cache_key = subfolder if subfolder else 'ALL'
    
    if cache_key not in _remote_files_cache:
        try:
            if Project.config.GS_PATH is None:
                raise RuntimeError('Globals not initialized. Call load_project(project_name) first.')
            gs_path: str = cast(str, Project.config.GS_PATH)
            
            # Build the full path to list
            if subfolder:
                gs_list_path = os.path.join(gs_path, subfolder)
            else:
                gs_list_path = gs_path
            
            import logging
            log = logging.getLogger(__name__)
            log.info(f"Fetching remote files from: {gs_list_path}")
            
            result = subprocess.run(
                ['gsutil', 'ls', '-r', gs_list_path],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            
            if result.returncode == 0:
                # Parse output to get all file paths
                files = set(
                    line.strip() 
                    for line in result.stdout.split('\n')
                    if line.strip() and not line.strip().endswith(':')
                )
                _remote_files_cache[cache_key] = files
                log.info(f"Loaded {len(files)} remote files from {cache_key}")
            else:
                # If command fails, return empty set
                log.warning(f"Failed to list remote files from {gs_list_path}: {result.stderr}")
                _remote_files_cache[cache_key] = set()
        except Exception as e:
            # If any error occurs, return empty set
            import logging
            log = logging.getLogger(__name__)
            log.error(f"Exception in get_remote_files: {e}")
            _remote_files_cache[cache_key] = set()
    
    return _remote_files_cache[cache_key]

def get_random_local_path() -> str:
    if LOCAL_OUTPUT_PATH is None:
        raise RuntimeError('Globals not initialized. Call load_project(project_name) first.')
    local_output_path: str = cast(str, LOCAL_OUTPUT_PATH)
    if CLUSTER_NAME == 'flame':
        return os.path.join(local_output_path, ''.join(random.choices(string.ascii_letters, k=8)))
    else:
        return local_output_path
