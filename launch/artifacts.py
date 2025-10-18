import os
import random
from dataclasses import dataclass
from experiments import Artifact, Task
from launch.globals import LOCAL_DATA_PATH, GS_PATH, OLMO_PATH, PROJECT_NAME, WANDB_ENTITY, GS_DATA_PATH    
from launch.utils.olmo_configuration import get_train_config
from launch.globals import get_remote_files, get_random_local_path
import math

LIST_OF_PRETRAIN_FILES = [
    f'datasets/c4/train/preprocessed_c4_v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix_gpt-neox-olmo-dolma-v1_5_part-{i:03d}-00000.npy'
    for i in range(0, 171)
]

LIST_OF_CPT_FILES= {
    "starcoder" : [
        "datasets/starcoder/train/preprocessed_starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00000.npy"
    ]
}

LIST_OF_VAL_FILES = {
    "c4-validation": ['datasets/c4/val/eval-data_perplexity_v3_small_gptneox20b_c4_en_val_part-0-00000.npy'],
    "starcoder-validation": ['datasets/starcoder/val/preprocessed_starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00001.npy']
}

def get_train_files(n_tokens, tokens_per_file=750_000_000):
    n_files = min(math.ceil(n_tokens*1024*1024*1024 / tokens_per_file) + 1, len(LIST_OF_PRETRAIN_FILES))
    return LIST_OF_PRETRAIN_FILES[:n_files]


@dataclass(frozen=True)
class PretrainedModel(Artifact):
    train_tokens: int
    optimizer: str = 'adamw'
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    sam_rho: float = 0.05
    batch_size: int = 256
    scheduler_name: str = 'cosine_with_warmup'
    scheduler_alpha_f: float = 0.1
    pretrain_gpus: int = 2
    
    @property
    def relpath(self) -> str:
        return f'PretrainedModel/{self.run_name}'

    @property
    def checkpoint_relpath(self) -> str:
        return f'{self.relpath}/final-unsharded'

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
    
    @property
    def exists(self) -> bool:
        remote_path = os.path.join(GS_PATH, self.checkpoint_relpath)
        remote_files = get_remote_files()
        return any(f.startswith(remote_path) for f in remote_files)
    
    def get_requirements(self):
        return {
            'gpus': self.pretrain_gpus,
            'nodes': 1,
            'cpus': self.pretrain_gpus * 2,
            'mem': '32GB',
            'requeue': True
        }
    
    def construct(self, builder: Task):
        local_data_path = get_random_local_path()

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
            f'torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d --nproc_per_node={self.pretrain_gpus} {train_script} {config_path}'
        )
        
        # Sync checkpoints from local to remote
        builder.rsync_to_gs(
            save_folder,
            remote_folder,
            delete=True,
            checksum=False,
            contents=True
        )


@dataclass(frozen=True)
class CPTModel(Artifact):
    train_tokens: int
    pretrained_model: PretrainedModel
    training_dataset_name: str
    optimizer: str = 'adamw'
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    batch_size: int = 64
    scheduler_name: str = 'cosine_with_warmup'
    scheduler_alpha_f: float = 0.1
    cpt_gpus: int = 1

    @property
    def relpath(self) -> str:
        return f'CPTModel/{self.training_dataset_name}/{self.run_name}'

    @property
    def checkpoint_relpath(self) -> str:
        return f'{self.relpath}/final-unsharded'

    @property
    def exists(self) -> bool:
        remote_path = os.path.join(GS_PATH, self.checkpoint_relpath)
        remote_files = get_remote_files()
        return any(f.startswith(remote_path) for f in remote_files)

    @property
    def run_name(self) -> str:
        pretrained_model_name = self.pretrained_model.run_name
        
        lr_str = f'{self.learning_rate:.0e}'.replace('e-0', 'e-')
        wd_str = f'{self.weight_decay:.0e}'.replace('e-0', 'e-') if self.weight_decay > 0 else '0'
        bs_str = str(self.batch_size)
        
        return f'{pretrained_model_name}-CPT-{self.training_dataset_name}-lr{lr_str}-wd{wd_str}-bs{bs_str}'

    def get_requirements(self):
        return {
            'gpus': self.cpt_gpus,
            'nodes': 1,
            'cpus': self.cpt_gpus * 2,
            'mem': '16GB',
        }

    def construct(self, builder: Task):
        local_data_path = get_random_local_path()
        run_name = self.run_name
        save_folder = os.path.join(local_data_path, self.relpath)
        remote_folder = os.path.join(GS_PATH, self.relpath)

        pretrained_model_relpath = self.pretrained_model.checkpoint_relpath
        pretrained_model_path = os.path.join(local_data_path, pretrained_model_relpath)
        
        builder.rsync_from_gs(
            os.path.join(GS_PATH, pretrained_model_relpath),
            pretrained_model_path,
            delete=True,
            checksum=True,
            skip_existing=False,
            check_exists=False,
            contents=True
        )

        # Build training data paths from the training_data artifact
        train_data_paths = [
            os.path.join(local_data_path, cpt_data_path)
            for cpt_data_path in LIST_OF_CPT_FILES[self.training_dataset_name]
        ]
        
        # Build eval datasets from validation_data
        eval_datasets = dict()
        for eval_dataset in LIST_OF_VAL_FILES:
            eval_datasets[eval_dataset] = [
                os.path.join(local_data_path, eval_data_path)
                for eval_data_path in LIST_OF_VAL_FILES[eval_dataset]
            ]

        config = get_train_config(
            run_name=run_name,
            save_folder=save_folder,
            train_data_paths=train_data_paths,
            eval_datasets=eval_datasets,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_duration=f'{self.train_tokens}e6T',
            stop_at=None,
            seed=6198,
            scheduler_name=self.scheduler_name,
            scheduler_alpha_f=self.scheduler_alpha_f,
            global_train_batch_size=self.batch_size,
            device_train_microbatch_size=4,
            eval_interval=100,
            save_interval_unsharded=1000,
            wandb_project=PROJECT_NAME,
            wandb_entity=WANDB_ENTITY,
            wandb_id=run_name,
            wandb_resume='allow',
            load_path=pretrained_model_path,
            reset_optimizer_state=True
        )

        # Ensure directory and save config
        builder.ensure_directory(save_folder)
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(config_path, config)

        # Collect all data paths
        all_data_paths = train_data_paths.copy()
        for eval_paths in eval_datasets.values():
            all_data_paths.extend(eval_paths)
        
        # Download directories from GS
        for local_dir in all_data_paths:
            gs_dir = local_dir.replace(local_data_path, GS_DATA_PATH)
            builder.download_from_gs(gs_dir, local_dir, directory=False)
        

        train_script = os.path.join(OLMO_PATH, 'scripts', 'train.py')
        builder.run_command(
            f'cd {OLMO_PATH} && '
            f'torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d --nproc_per_node={self.cpt_gpus} {train_script} {config_path}'
        )

        # Upload the finetuned model to remote storage
        builder.upload_to_gs(
            save_folder,
            remote_folder,
            directory=True,
        )


@dataclass(frozen=True)
class ModelEvaluation(Artifact):
    model: PretrainedModel | CPTModel
    device: str = 'cuda'
    chunk_size: int = 1024

    @property
    def relpath(self) -> str:
        # Place results in ModelEvaluation folder, filename derived from model run name
        return f'ModelEvaluation/{self.model.run_name}-eval.json'

    @property
    def exists(self) -> bool:
        remote_path = os.path.join(GS_PATH, self.relpath)
        remote_files = get_remote_files()
        return any(f == remote_path for f in remote_files)

    def get_requirements(self):
        return {
            'gpus': 1,
            'nodes': 1,
            'cpus': 2,
            'mem': '16GB',
            'requeue': True
        }

    def construct(self, builder: Task):
        # Create a unique local working directory to avoid collisions
        
        local_root = get_random_local_path()

        # Prepare local output path and ensure directory exists
        local_output_path = os.path.join(local_root, self.relpath)
        builder.ensure_directory(os.path.dirname(local_output_path))

        # Ensure model checkpoint is available locally (rsync from GS)
        local_model_path = os.path.join(local_root, self.model.checkpoint_relpath)
        remote_model_path = os.path.join(GS_PATH, self.model.checkpoint_relpath)
        builder.rsync_from_gs(
            remote_model_path,
            local_model_path,
            delete=False,
            checksum=False,
            skip_existing=False,
            check_exists=True,
            contents=True
        )

        # Collect and download evaluation data files
        local_eval_paths = []
        for eval_paths in LIST_OF_VAL_FILES.values():
            for eval_path in eval_paths:
                local_eval_path = os.path.join(local_root, eval_path)
                local_eval_paths.append(local_eval_path)
        # Download each eval file from GS_DATA_PATH mirror
        for local_path in local_eval_paths:
            gs_path = local_path.replace(local_root, GS_DATA_PATH)
            builder.download_from_gs(gs_path, local_path, directory=False)

        # Build comma-separated list for evaluator
        data_path_arg = ','.join(local_eval_paths)

        # Path to evaluation script
        eval_script = os.path.join('scripts', 'evaluate.py')

        # Run evaluation
        builder.run_command(
            ' '.join([
                'python', eval_script,
                f"--model_path {local_model_path}",
                f"--device {self.device}",
                f"--data_path {data_path_arg}",
                f"--chunk_size {self.chunk_size}",
                f"--output_path {local_output_path}",
            ])
        )

        # Upload results to GS
        local_eval_dir = os.path.dirname(local_output_path)
        remote_eval_dir = os.path.join(GS_PATH, os.path.dirname(self.relpath))
        builder.rsync_to_gs(
            local_eval_dir,
            remote_eval_dir,
            delete=False,
            checksum=False,
            contents=True
        )
