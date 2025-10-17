"""Pre-training artifacts for the Wikipedia pretraining experiment."""

import os
import hashlib
from dataclasses import dataclass
import math
import random

from experiments import Artifact, ArtifactSet, Task

from launch.utils.olmo_configuration import get_train_config

from launch.globals import (
    LOCAL_DATA_PATH,
    GS_PATH,
    OLMO_PATH,
    PROJECT_NAME,
    WANDB_ENTITY,
    GS_DATA_PATH
)

LIST_OF_TRAIN_FILES = [
    f'datasets/c4/train/preprocessed_c4_v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix_gpt-neox-olmo-dolma-v1_5_part-{i:03d}-00000.npy'
    for i in range(0, 171)
]
LIST_OF_VAL_FILES = {
    "c4-validation": ['datasets/c4/val/eval-data_perplexity_v3_small_gptneox20b_c4_en_val_part-0-00000.npy']
}

def get_train_files(n_tokens, tokens_per_file=750_000_000):
    n_files = min(math.ceil(n_tokens*1024*1024*1024 / tokens_per_file) + 1, len(LIST_OF_TRAIN_FILES))
    return LIST_OF_TRAIN_FILES[:n_files]

PRETRAIN_GPUS = 2

@dataclass(frozen=True)
class PretrainedModel(Artifact):
    train_tokens: int
    optimizer: str = 'adamw'
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    sam_rho: float = 0.05
    batch_size: int = 256
    scheduler_name: str = 'cosine_with_warmup'
    scheduler_t_warmup: int = 100
    scheduler_alpha_f: float = 0.1
    
    @property
    def relpath(self) -> str:
        return f'PretrainedModel/{self.run_name}'

    @property
    def run_name(self) -> str:
        lr_str = f'{self.learning_rate:.0e}'.replace('e-0', 'e-')
        wd_str = f'{self.weight_decay:.0e}'.replace('e-0', 'e-') if self.weight_decay > 0 else '0'
        bs_str = str(self.batch_size)
        tk_str = f'{self.train_tokens}B'

        final_run_name = f'OLMo-tk{tk_str}-{self.optimizer}-lr{lr_str}-wd{wd_str}-bs{bs_str}'
        if self.optimizer == 'sam':
            final_run_name += f'-rho{self.sam_rho:.0e}'.replace('e-0', 'e-')

        return final_run_name 
    
    # @property
    # def exists(self) -> bool:
    #     # Check local existence
    #     local_path = os.path.join(LOCAL_DATA_PATH, self.relpath)
    #     if os.path.exists(local_path) and os.path.isdir(local_path):
    #         return True
        
    #     # Check remote existence if enabled
    #     if CHECK_EXISTS_REMOTE:
    #         remote_path = os.path.join(GS_PATH, self.relpath)
    #         remote_files = get_remote_files()
    #         # Check if any file in the remote list starts with our remote path
    #         # This is because a directory will have multiple files under it
    #         return any(f.startswith(remote_path) for f in remote_files)
        
    #     return False
    
    def get_requirements(self):
        return {
            'gpus': PRETRAIN_GPUS,
            'nodes': 1,
            'cpus': PRETRAIN_GPUS * 2,
            'mem': '32GB',
            'requeue': True
        }
    
    def construct(self, builder: Task):
        random_string = ''.join([random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') for i in range(8)])
        local_data_path = os.path.join(LOCAL_DATA_PATH, random_string)

        run_name = self.run_name
        save_folder = os.path.join(local_data_path, self.relpath)
        remote_folder = os.path.join(GS_PATH, self.relpath)
        
        # Build training data paths from the training_data artifact set
        train_data_paths = []
        for train_data_path in get_train_files(self.train_tokens):
            train_data_paths.append(
                os.path.join(local_data_path, train_data_path)
            )
        
        # Build eval datasets from validation_data
        eval_datasets = dict()
        for eval_dataset in LIST_OF_VAL_FILES:
            eval_datasets[eval_dataset] = [
                os.path.join(local_data_path, eval_data_path)
                for eval_data_path in LIST_OF_VAL_FILES[eval_dataset]
            ]
        
        # Create pretrain config using configuration utility
        config = get_train_config(
            run_name=run_name,
            save_folder=save_folder,
            train_data_paths=train_data_paths,
            eval_datasets=eval_datasets,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            sam_rho=self.sam_rho,
            max_duration=f"{self.train_tokens}e9T",
            stop_at=None,
            seed=6198,
            scheduler_name=self.scheduler_name,
            scheduler_alpha_f=self.scheduler_alpha_f,
            global_train_batch_size=self.batch_size,
            device_train_microbatch_size=4,
            eval_interval=1000,
            save_interval_unsharded=5000,
            wandb_project=PROJECT_NAME,
            wandb_entity=WANDB_ENTITY,
            wandb_id=run_name,
            wandb_resume='allow',
            try_load_latest_save=True,
            run_sync_cmd=True,
        )
        
        # Set SYNC_CMD environment variable
        builder.set_env("SYNC_CMD", f"gsutil -m rsync -r {save_folder}/ {remote_folder}/")
        
        # Ensure directory and save config
        builder.ensure_directory(save_folder)
        config_path = os.path.join(save_folder, 'config.yaml')

        # Sync checkpoints from remote to local (if exists)
        builder.rsync_from_gs(
            remote_folder,
            save_folder,
            delete=True,
            checksum=False,
            skip_existing=False,
            check_exists=True,  # Since the directory is already created above
            contents=True
        )
        
        builder.create_yaml_file(config_path, config)

        # Collect all data paths
        all_data_paths = train_data_paths.copy()
        for eval_paths in eval_datasets.values():
            all_data_paths.extend(eval_paths)
        
        # Download directories from GS
        for local_dir in all_data_paths:
            gs_dir = local_dir.replace(local_data_path, GS_DATA_PATH)
            builder.download_from_gs(gs_dir, local_dir, directory=False)
        
        # Setup OLMo environment and run training
        train_script = os.path.join(OLMO_PATH, 'scripts', 'train.py')
        builder.run_command(
            f'cd {OLMO_PATH} && '
            f'torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d --nproc_per_node={PRETRAIN_GPUS} {train_script} {config_path}'
        )
        
        # Sync checkpoints from local to remote
        builder.rsync_to_gs(
            save_folder,
            remote_folder,
            delete=True,
            checksum=False,
            contents=True
        )

sam_pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['sam'],
            'learning_rate': [5e-4, 1e-3, 5e-3, 1e-2],
            'train_tokens': [1, 4],
            'weight_decay': [0.02],
            'batch_size': [256],
            'scheduler_name': ['cosine_with_warmup'],
            'sam_rho': [0.05, 0.1, 0.2],
            # 'scheduler_t_warmup': [100],
            'scheduler_alpha_f': [0.1],
        }
    )

sgd_pretrained_models = ArtifactSet.from_product(
        cls=PretrainedModel,
        params={
            'optimizer': ['sgd'],
            'learning_rate': [5e-4, 1e-3, 5e-3, 1e-2],
            'train_tokens': [1, 4],
            'weight_decay': [0.02, 0.005],
            'batch_size': [256],
            'scheduler_name': ['cosine_with_warmup'],
            # 'scheduler_t_warmup': [100],
            'scheduler_alpha_f': [0.1],
        }
    )