import os
import random
from dataclasses import dataclass
from typing import cast
from experiments import Artifact, Task, Project  # type: ignore
from launch import globals as G
from launch.utils.olmo_configuration import get_train_config
import math
import logging

log = logging.getLogger(__name__)

LIST_OF_PRETRAIN_FILES = {
    "c4": {
        "data_paths" : [f'c4/train/preprocessed_c4_v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix_gpt-neox-olmo-dolma-v1_5_part-{i:03d}-00000.npy' for i in range(0, 171)],
        "tokens_per_file" : 750_000_000,
        "val" : {
            "c4-validation": ['c4/val/eval-data_perplexity_v3_small_gptneox20b_c4_en_val_part-0-00000.npy'],
        }
    },
    "dclm" : {
        "data_paths" : [
            f'dclm/train/preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer_part-{i:03d}/{j:05d}.npy'
            for i in range(0, 60)
            for j in range(0, 5) 
        ],
        "tokens_per_file" : 3_000_000_000,
        "val" : {
            "dclm-validation": ['dclm/val/preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer_part-187-00004-2M.npy'],
            "c4-validation": ['c4/val/eval-data_perplexity_v3_small_dolma2-tokenizer_c4_en_val_part-0-00000.npy'],
        }
    }
}

LIST_OF_CPT_FILES= {
    "c4" : {
        "starcoder" : {
            "data_paths" : [
                "starcoder/train/preprocessed_starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00000.npy"
            ],
            "val" : {
                "starcoder-validation": ['starcoder/val/preprocessed_starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00001.npy'],
            }
        },
    },
    "dclm" : {
        "starcoder" : {
            "data_paths" : [
                "starcoder/train/preprocessed_starcoder_v1-decon-100_to_20k-2star-top_token_030_allenai_dolma2-tokenizer_part-000-00000.npy"
            ],
            "val" : {
                "starcoder-validation": ['starcoder/val/preprocessed_starcoder_v1-decon-100_to_20k-2star-top_token_030_allenai_dolma2-tokenizer_part-001-00000-2M.npy'],
            }
        },
        "proofpile" : {
            "data_paths" : [
                "proofpile/train/preprocessed_proof-pile-2_v0_decontaminated_arxiv_train_allenai_dolma2-tokenizer_part-00-00000.npy"
            ],
            "val" : {
                "proofpile-validation": ['proofpile/val/preprocessed_proof-pile-2_v0_decontaminated_arxiv_train_allenai_dolma2-tokenizer_part-01-00000-2M.npy'],
            }
        }
    }
}

for train_dataset in LIST_OF_PRETRAIN_FILES.keys():

    val_dataset = LIST_OF_PRETRAIN_FILES[train_dataset]["val"]
    cpt_datasets = LIST_OF_CPT_FILES[train_dataset]

    for cpt_dataset in cpt_datasets.keys():
        cpt_datasets[cpt_dataset]["val"].update(val_dataset)

    LIST_OF_CPT_FILES[train_dataset] = cpt_datasets

BILLION = 1024**3
MILLION = 1024**2

def get_train_files(dataset_name, n_tokens):

    dataset_info = LIST_OF_PRETRAIN_FILES[dataset_name]
    tokens_per_file = dataset_info["tokens_per_file"]
    data_paths = dataset_info["data_paths"]

    n_files = min(math.ceil(n_tokens*BILLION / tokens_per_file) + 1, len(data_paths))
    log.info(f"Downloading {n_files} files from {dataset_name.upper()}")
    return data_paths[:n_files]


@dataclass(frozen=True)
class PretrainedModel(Artifact):
    train_tokens: int
    model_size: str = '20m'
    optimizer: str = 'adamw'
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    sam_rho: float = 0.05
    sam_base_optimizer: str = 'adamw'
    batch_size: int = 256
    scheduler_name: str = 'cosine_with_warmup'
    scheduler_alpha_f: float = 0.1
    pretrain_gpus: int = 8
    muon_learning_rate: float = 5e-1 #EDIT
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    train_dataset: str = "dclm"
 
    
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
            final_run_name = f'OLMo-tk{tk_str}-{self.optimizer}_{self.sam_base_optimizer}-lr{lr_str}-wd{wd_str}-bs{bs_str}'
            final_run_name += f'-rho{self.sam_rho:.0e}'.replace('e-0', 'e-')
        if self.optimizer =='muon':
            final_run_name += f'-muon_lr{self.muon_learning_rate:.0e}'.replace('e-0', 'e-')

        return final_run_name 
    
    @property
    def exists(self) -> bool:
        gs_path: str = cast(str, Project.config.GS_PATH)
        remote_path = os.path.join(gs_path, self.checkpoint_relpath)
        remote_files = G.get_remote_files()
        return any(f.startswith(remote_path) for f in remote_files)
    
    def get_requirements(self):
        return {
            # 'gpus': self.pretrain_gpus,
            "gres": f"gpu:{self.pretrain_gpus}",
            'nodes': 1,
            'cpus': max(1, self.pretrain_gpus * 2),
            'mem': '64GB',
            'requeue': True,
            'partition': 'flame',
            'qos': 'flame-16gpu_qos',
            'account': 'aditirag',
            "time": "2-00:00:00"
        }
    
    def construct(self, builder: Task):
        local_output_path = G.get_random_local_path()
        local_data_path = G.LOCAL_DATA_PATH

        if G.DOWNLOAD_DATA:
            local_data_path = local_output_path
            
        run_name = self.run_name
        save_folder = os.path.join(local_output_path, self.relpath)
        gs_path: str = cast(str, G.GS_PATH)
        remote_folder = os.path.join(gs_path, self.relpath)
        
        # Build training data paths from the training_data artifact set
        train_data_paths = []
        for train_data_path in get_train_files(self.train_dataset, self.train_tokens):
            train_data_paths.append(
                os.path.join(local_data_path, train_data_path)
            )
        
        # Build eval datasets from validation_data
        eval_datasets = dict()
        for eval_dataset in LIST_OF_PRETRAIN_FILES[self.train_dataset]["val"]:
            eval_datasets[eval_dataset] = [
                os.path.join(local_data_path, eval_data_path)
                for eval_data_path in LIST_OF_PRETRAIN_FILES[self.train_dataset]["val"][eval_dataset]
            ]
        
        # Create pretrain config using configuration utility
        project_name: str = cast(str, G.PROJECT_NAME)
        wandb_entity: str = cast(str, G.WANDB_ENTITY)

        model_overrides = dict()
        if self.train_dataset == "dclm":
            model_overrides = {
                'vocab_size': 100278,
                'embedding_size': 100352,
                'eos_token_id': 100257,
                'pad_token_id': 100277,
            }

        config = get_train_config(
            run_name=run_name,
            save_folder=save_folder,
            train_data_paths=train_data_paths,
            eval_datasets=eval_datasets,
            model_size=self.model_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            muon_learning_rate = self.muon_learning_rate,
            muon_momentum = self.muon_momentum,
            muon_weight_decay = self.muon_weight_decay,
            sam_rho=self.sam_rho,
            sam_base_optimizer=self.sam_base_optimizer,
            max_duration=f"{self.train_tokens}e9T",
            stop_at=None,
            seed=6198,
            model_overrides=model_overrides,
            scheduler_name=self.scheduler_name,
            scheduler_alpha_f=self.scheduler_alpha_f,
            global_train_batch_size=self.batch_size,
            device_train_microbatch_size=8,
            eval_interval=5000,
            save_interval_unsharded=5000,
            wandb_project=project_name,
            wandb_entity=wandb_entity,
            wandb_id=run_name,
            wandb_resume='allow',
            try_load_latest_save=True,
            run_sync_cmd=True,
            tokenizer={
                'identifier': 'tokenizers/allenai_dolma2.json' if self.train_dataset=='dclm' else 'tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json',
                'truncate_direction': 'right',
            }
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
        if G.DOWNLOAD_DATA:
            # Extract unique parent directories to download
            parent_dirs = set()
            
            for local_file_path in all_data_paths:
                parent_dir = os.path.dirname(local_file_path)
                parent_dirs.add(parent_dir)

            print(parent_dirs)
            
            # Download each unique parent directory
            gs_data_path: str = cast(str, G.GS_DATA_PATH)
            for local_dir in parent_dirs:
                gs_dir = local_dir.replace(local_data_path, gs_data_path)
                builder.download_from_gs(gs_dir, local_dir, directory=True)
        
        # Setup OLMo environment and run training
        olmo_path: str = cast(str, Project.config.OLMO_PATH)
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && '
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
        gs_root: str = cast(str, Project.config.GS_PATH)
        remote_path = os.path.join(gs_root, self.checkpoint_relpath)
        remote_files = G.get_remote_files()
        return any(f.startswith(remote_path) for f in remote_files)

    @property
    def run_name(self) -> str:
        pretrained_model_name = self.pretrained_model.run_name

        #EDITED FOR MUON
        
        lr_str = f'{self.learning_rate:.0e}'.replace('e-0', 'e-')
        wd_str = f'{self.weight_decay:.0e}'.replace('e-0', 'e-') if self.weight_decay > 0 else '0'
        bs_str = str(self.batch_size)
        
        return f'{pretrained_model_name}-CPT-{self.training_dataset_name}-lr{lr_str}-wd{wd_str}-bs{bs_str}'

    def get_requirements(self):
        return {
            'gpus': self.cpt_gpus,
            'nodes': 1,
            'cpus': self.cpt_gpus * 2,
            'mem': '64GB',
            'gres' : "gpu:1"
        }

    def construct(self, builder: Task):
        local_data_path = G.get_random_local_path()
        run_name = self.run_name
        save_folder = os.path.join(local_data_path, self.relpath)
        gs_root: str = cast(str, Project.config.GS_PATH)
        remote_folder = os.path.join(gs_root, self.relpath)

        pretrained_model_relpath = self.pretrained_model.checkpoint_relpath
        pretrained_model_path = os.path.join(local_data_path, pretrained_model_relpath)
        
        builder.rsync_from_gs(
            os.path.join(gs_root, pretrained_model_relpath),
            pretrained_model_path,
            delete=True,
            checksum=True,
            skip_existing=False,
            check_exists=False,
            contents=True
        )

        # Build training data paths from the training_data artifact
        train_data_paths = []
        for cpt_data_path in LIST_OF_CPT_FILES[self.pretrained_model.train_dataset][self.training_dataset_name]["data_paths"]:
            train_data_paths.append(os.path.join(local_data_path, cpt_data_path))
        
        # Build eval datasets from validation_data
        eval_datasets = dict()
        for eval_dataset in LIST_OF_CPT_FILES[self.pretrained_model.train_dataset][self.training_dataset_name]["val"]:
            eval_datasets[eval_dataset] = [
                os.path.join(local_data_path, eval_data_path)
                for eval_data_path in LIST_OF_CPT_FILES[self.pretrained_model.train_dataset][self.training_dataset_name]["val"][eval_dataset]
            ]

        project_name: str = cast(str, Project.config.PROJECT_NAME)
        wandb_entity: str = cast(str, Project.config.WANDB_ENTITY)
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
            wandb_project=project_name,
            wandb_entity=wandb_entity,
            wandb_id=run_name,
            wandb_resume='allow',
            load_path=pretrained_model_path,
            reset_optimizer_state=True,
            tokenizer='tokenizers/allenai_dolma2.json' if self.pretrained_model.train_dataset=='dclm' else 'tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json'
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
            gs_data_path: str = cast(str, Project.config.GS_DATA_PATH)
            gs_dir = local_dir.replace(local_data_path, gs_data_path)
            builder.download_from_gs(gs_dir, local_dir, directory=False)
        

        olmo_path: str = cast(str, Project.config.OLMO_PATH)
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && '
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
    model: "PretrainedModel | CPTModel"
    device: str = 'cuda'
    chunk_size: int = 1024

    @property
    def relpath(self) -> str:
        # Place results in ModelEvaluation folder, filename derived from model run name
        return f'ModelEvaluation/{self.model.run_name}-eval.json'

    @property
    def exists(self) -> bool:
        gs_path: str = cast(str, Project.config.GS_PATH)
        remote_path = os.path.join(gs_path, self.relpath)
        remote_files = G.get_remote_files()
        return any(f == remote_path for f in remote_files)

    def get_requirements(self):
        return {
            'gpus':1,
            'nodes': 1,
            'cpus': 2,
            'mem': '16GB',
            'requeue': True
        }

    def construct(self, builder: Task):
        # Create a unique local working directory to avoid collisions
        
        local_root = G.get_random_local_path()

        # Prepare local output path and ensure directory exists
        local_output_path = os.path.join(local_root, self.relpath)
        builder.ensure_directory(os.path.dirname(local_output_path))

        # Ensure model checkpoint is available locally (rsync from GS)
        local_model_path = os.path.join(local_root, self.model.checkpoint_relpath)
        gs_root: str = cast(str, Project.config.GS_PATH)
        remote_model_path = os.path.join(gs_root, self.model.checkpoint_relpath)
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
        if isinstance(self.model, PretrainedModel):
            for eval_dataset in LIST_OF_PRETRAIN_FILES[self.model.train_dataset]["val"]:
                for eval_path in LIST_OF_PRETRAIN_FILES[self.model.train_dataset]["val"][eval_dataset]:
                    local_eval_path = os.path.join(local_root, eval_path)
                    local_eval_paths.append(local_eval_path)
        elif isinstance(self.model, CPTModel):
            for eval_dataset in LIST_OF_CPT_FILES[self.model.pretrained_model.train_dataset][self.model.training_dataset_name]["val"]:
                for eval_path in LIST_OF_CPT_FILES[self.model.pretrained_model.train_dataset][self.model.training_dataset_name]["val"][eval_dataset]:
                    local_eval_path = os.path.join(local_root, eval_path)
                    local_eval_paths.append(local_eval_path)

        # Download each eval file from GS_DATA_PATH mirror
        for local_path in local_eval_paths:
            gs_data_path: str = cast(str, Project.config.GS_DATA_PATH)
            gs_path_var = local_path.replace(local_root, gs_data_path)
            builder.download_from_gs(gs_path_var, local_path, directory=False)

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
        gs_path: str = cast(str, Project.config.GS_PATH)
        remote_eval_dir = os.path.join(gs_path, os.path.dirname(self.relpath))
        builder.rsync_to_gs(
            local_eval_dir,
            remote_eval_dir,
            delete=False,
            checksum=False,
            contents=True
        )
