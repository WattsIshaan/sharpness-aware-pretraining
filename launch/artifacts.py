import os
import random
import math
import json
import logging
from dataclasses import dataclass
from typing import cast, Dict, List, Any

from experiments import Artifact, Task, Project  # type: ignore
from launch import globals as G
from launch.utils.olmo_configuration import get_train_config

log = logging.getLogger(__name__)

BILLION = 10**9
MILLION = 10**6


LIST_OF_PRETRAIN_FILES = {
    "dclm": {
        "data_paths": [
            f'dclm/train/preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer_part-{i:03d}/{j:05d}.npy'
            for i in range(50) for j in range(5)
        ],
        "anneal_data_paths": [
            f'dclm/train/preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer_part-{i:03d}/{j:05d}.npy'
            for i in range(50, 60) for j in range(5)
        ],
        "tokens_per_file": 3 * BILLION,
        "val": {
            "dclm-validation": ['dclm/val/dclm-20m.npy'],
        }
    },
}

# SFT datasets per pretraining corpus: tokenized inputs, label masks, and val sets.
LIST_OF_SFT_FILES = {
    "dclm": {
        "tulu": {
            "data_paths": ["allenai_tulu-3-sft-mixture/train/input_ids.npy"],
            "mask_paths": ["allenai_tulu-3-sft-mixture/train/label_mask.npy"],
            "val": {"tulu-validation": {"data": ['allenai_tulu-3-sft-mixture/val/input_ids-tulu.npy'], "masks": ['allenai_tulu-3-sft-mixture/val/label_mask.npy']}}
        },
        "starcoder": {
            "data_paths": ["bigcode_starcoderdata/train/input_ids.npy"],
            "mask_paths": ["bigcode_starcoderdata/train/label_mask.npy"],
            "val": {"starcoder-validation": {"data": ['bigcode_starcoderdata/val/input_ids-starcoder.npy'], "masks": ['bigcode_starcoderdata/val/label_mask.npy']}}
        },
        "musicpile": {
            "data_paths": ["musicpile/train/input_ids.npy"],
            "mask_paths": ["musicpile/train/label_mask.npy"],
            "val": {"musicpile-validation": {"data": ['musicpile/val/input_ids-musicpile.npy'], "masks": ['musicpile/val/label_mask.npy']}}
        },
        "gsm8k": {
            "data_paths": ["openai_gsm8k/train/input_ids.npy"],
            "mask_paths": ["openai_gsm8k/train/label_mask.npy"],
            "val": {"gsm8k-validation": {"data": ['openai_gsm8k/val/input_ids-gsm8k.npy'], "masks": ['openai_gsm8k/val/label_mask.npy']}},
            "train_tokens": 1.2,
        },
        "stackmathqa": {
            "data_paths": ["math-ai_StackMathQA/train/input_ids.npy"],
            "mask_paths": ["math-ai_StackMathQA/train/label_mask.npy"],
            "val": {"stackmathqa-validation": {"data": ['math-ai_StackMathQA/val/input_ids-stackmathqa.npy'], "masks": ['math-ai_StackMathQA/val/label_mask.npy']}},
        },

    },
    "dolmino": {
        "tulu": {
            "data_paths": ["allenai_tulu-3-sft-mixture/train/input_ids.npy"],
            "mask_paths": ["allenai_tulu-3-sft-mixture/train/label_mask.npy"],
            "val": {"tulu-validation": {"data": ['allenai_tulu-3-sft-mixture/val/input_ids-tulu.npy'], "masks": ['allenai_tulu-3-sft-mixture/val/label_mask.npy']}},
        },
        "musicpile": {
            "data_paths": ["musicpile/train/input_ids.npy"],
            "mask_paths": ["musicpile/train/label_mask.npy"],
            "val": {"musicpile-validation": {"data": ['musicpile/val/input_ids-musicpile.npy'], "masks": ['musicpile/val/label_mask.npy']}},
        },
        "stackmathqa": {
            "data_paths": ["math-ai_StackMathQA/train/input_ids.npy"],
            "mask_paths": ["math-ai_StackMathQA/train/label_mask.npy"],
            "val": {"stackmathqa-validation": {"data": ['math-ai_StackMathQA/val/input_ids-stackmathqa.npy'], "masks": ['math-ai_StackMathQA/val/label_mask.npy']}},
        },
        "meta-math": {
            "data_paths": ["meta-math_MetaMathQA/train/input_ids.npy"],
            "mask_paths": ["meta-math_MetaMathQA/train/label_mask.npy"],
            "val": {"meta-math-validation": {"data": ['meta-math_MetaMathQA/val/input_ids-metamath.npy'], "masks": ['meta-math_MetaMathQA/val/label_mask.npy']}},
        },
    },
}

# Budget-matched peak LR, keyed by [model_size_millions][train_tokens_billions].
# Used when PretrainedModel.pt_lr == -1.
PT_LR = {
    20 : {
        4: 3e-3,
        8: 3e-3,
        16: 3e-3,
        32: 1e-3,
        64: 1e-3,
    },
    60: {
        12: 1e-3,
        24: 1e-3,
        48: 6e-4,
        96: 6e-4,
        192: 3e-4,
    },
    150: {
        15: 1e-3,
        30: 1e-3,
        60: 6e-4,
        120: 3e-4,
        240: 3e-4,
    },
}

# Warmup steps keyed by model size (millions of params).
WARMUP_STEPS = {
    20: 1000,
    60: 2000,
    150: 3000,
}

# Checkpoint step to anneal from, keyed by [anneal_percent][match][model_size][pt_token].
# "match" = whether the anneal budget is matched by token count or by compute.
ANNEAL_CKPT = {
    5: {
        "compute": {},
        "token": {
            20: {
                64: 230000,
            },
            60: {
                192: 700000,
            },
            150: {
                120: 435000,
            }
        },
    },
    10: {    
        "compute": {
            20: {
                4: 15000,
                8: 25000,
                16: 50000,
                32: 100000,
                64: 205000,
            },
            60:{
                12: 35000,
                24: 75000,
                48: 155000,
                96: 305000,
                192: 610000,
            },
            150 : {
                15: 45000,
                30: 95000,
                60: 190000,
                120: 380000,
                240: 760000,
            }
        },
        "token": {
            20: {
                4: 15000,
                8: 30000,
                16: 55000,
                32: 110000,
                64: 220000,
            },
            60:{
                12: 40000,
                24: 85000,
                48: 165000,
                96: 335000,
                192: 670000,
            },
            150 : {
                15: 50000,
                30: 105000,
                60: 205000,
                120: 415000,
                240: 830000,
            }
        }
    },
    20: {
        "compute": {},
        "token": {
            20: {
                64: 205000,
            },
            60: {
                192: 610000,
            },
            150: {
                120: 380000,
            }
        },
    },
}

# WSD anneal-begin step for AnnealedModel2, keyed by [model_size][anneal_percent].
ANNEAL_CKPT2 = {
    60: {
        5: 695000,
        10: 660000,
        20: 585000,
        50: 365000,
        100: 0,
    },
}

# Add each corpus's pretrain validation set to every SFT dataset so SFT runs also
# track pretrain (forgetting) perplexity alongside the SFT validation set.
for base in LIST_OF_PRETRAIN_FILES:
    base_val = LIST_OF_PRETRAIN_FILES[base]["val"]
    for sft in LIST_OF_SFT_FILES.get(base, {}):
        LIST_OF_SFT_FILES[base][sft]["val"].update(base_val)


def pretrain_train_memmap_paths_for_ewc_fisher_subsample(
    pretrain_dataset: str,
    local_root: str,
    subsample_tokens_billion: float,
) -> List[str]:
    """
    Local paths to a **prefix** of pretrain **training** memmaps (``data_paths`` / cosine main stage)
    for EWC Fisher estimation. File count matches ``get_train_files``-style logic.
    """
    if pretrain_dataset not in LIST_OF_PRETRAIN_FILES:
        raise ValueError(
            f"EWC Fisher on pretrain train requires {pretrain_dataset!r} in LIST_OF_PRETRAIN_FILES "
            "(or pass ewc_fisher_paths manually in config)."
        )
    if subsample_tokens_billion <= 0:
        raise ValueError("subsample_tokens_billion must be positive")

    info = LIST_OF_PRETRAIN_FILES[pretrain_dataset]
    data_source = info["data_paths"]
    n_files = min(
        math.ceil(subsample_tokens_billion * BILLION / info["tokens_per_file"]) + 1,
        len(data_source),
    )
    n_files = max(1, n_files)
    log.info(
        "EWC Fisher pretrain subsample: dataset=%s target≈%.5gB tokens → %d/%d train memmap files",
        pretrain_dataset,
        subsample_tokens_billion,
        n_files,
        len(data_source),
    )
    return [os.path.join(local_root, p) for p in data_source[:n_files]]



def get_train_files(dataset_name: str, n_tokens_billion: int, train_stage: str = 'stable') -> List[str]:
    # Return just enough shards to cover the requested token budget (+1 for slack).
    # 'stable' uses the main training shards; otherwise the held-out decay-stage shards.
    info = LIST_OF_PRETRAIN_FILES[dataset_name]
    data_source = info["data_paths"] if train_stage == 'stable' else info["anneal_data_paths"]
    
    n_files = min(math.ceil(n_tokens_billion * BILLION / info["tokens_per_file"]) + 1, len(data_source))
    log.info(f"Downloading {n_files} files from {dataset_name.upper()}")
    return data_source[:n_files]


# --- Artifact classes: one per pipeline stage ---

@dataclass(frozen=True)
class PretrainedModel(Artifact):
    """A from-scratch pretraining run (stage-1).

    Trains an OLMo2 model of ``model_size`` on ``pretrain_dataset`` for
    ``train_tokens`` billion tokens and syncs the final checkpoint to GCS.
    Downstream artifacts (anneal, SFT, perturb, eval) load this checkpoint.
    """
    train_tokens: int                            # pretraining budget, in billions of tokens
    model_size: str = '20m'                      # config preset ('20m', '60m', '150m', '300m')
    optimizer: str = 'adamw'                     # 'adamw' or 'sam'
    pt_lr: float = 6e-4                           # peak LR; -1 auto-selects the budget-matched LR from PT_LR
    weight_decay: float = 0.1
    sam_rho: float = 0.05                         # SAM neighborhood size (only used when optimizer == 'sam')
    sam_base_optimizer: str = 'adamw'            # inner optimizer wrapped by SAM
    batch_size: int = 256                        # global train batch size
    scheduler_name: str = 'cosine_with_warmup'   # LR schedule
    scheduler_alpha_f: float = 0.1               # final-LR fraction (eta_min = alpha_f * peak_lr)
    pretrain_gpus: int = 8                        # GPUs (single node)
    pretrain_dataset: str = "dclm"
    sequence_length: int = 1024

    @property
    def learning_rate(self) -> float:
        # pt_lr == -1 => auto-select the budget-matched peak LR for this size/token count.
        if self.pt_lr == -1:
            return PT_LR[int(self.model_size[:-1])][self.train_tokens]
        return self.pt_lr

    @property
    def scheduler_t_warmup(self) -> int:
        # Warmup steps are fixed per model size.
        return WARMUP_STEPS[int(self.model_size[:-1])]

    @property
    def relpath(self) -> str:
        # GCS/local subpath where this run's artifacts live.
        return f'PretrainedModel/{self.run_name}'

    @property
    def checkpoint_relpath(self) -> str:
        # Final unsharded checkpoint consumed by downstream artifacts.
        return f'{self.relpath}/final-unsharded'

    @property
    def run_name(self) -> str:
        # Unique id; also used as the wandb id and GCS folder name. SAM encodes base optimizer + rho.
        lr_str = f'{self.learning_rate:.0e}'.replace('e-0', 'e-')
        wd_str = f'{self.weight_decay:.0e}'.replace('e-0', 'e-') if self.weight_decay > 0 else '0'
        tk_str = f'{self.train_tokens}B'
        
        name = f'OLMo2-{self.model_size}-tk{tk_str}-{self.optimizer}-lr{lr_str}-wd{wd_str}-bs{self.batch_size}'
        
        if self.optimizer == 'sam':
            name = f'OLMo2-{self.model_size}-tk{tk_str}-sam_{self.sam_base_optimizer}-lr{lr_str}-wd{wd_str}-bs{self.batch_size}'
            name += f'-rho{self.sam_rho:.0e}'.replace('e-0', 'e-')
            
        return name

    @property
    def exists(self) -> bool:
        # Skip the job if the final checkpoint is already on GCS (unless checking is disabled).
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
            
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder='PretrainedModel')
        
        found = any(f.startswith(remote_path) for f in remote_files)
        status = "✓ EXISTS" if found else "❌ NOT found"
        log.info(f"[PretrainedModel] {status}: {self.run_name}")
        return found

    def get_requirements(self) -> Dict[str, Any]:
        # Slurm resource request for the pretraining job.
        return {
            "gpus": self.pretrain_gpus,
            "nodes": 1,
            "cpus": self.pretrain_gpus * 2,
            "mem": '64GB',
            "requeue": True,
            "time": "2-00:00:00",
            "partition": 'general',
        }

    def construct(self, builder: Task):
        # Build the OLMo train config, stage data/checkpoints, launch torchrun, sync to GCS.
        local_output_path = G.get_random_local_path()
        # Data is already local unless DOWNLOAD_DATA is set (then it is pulled from GCS below).
        local_data_path = local_output_path if G.DOWNLOAD_DATA else G.LOCAL_DATA_PATH
        assert local_data_path is not None, "LOCAL_DATA_PATH must be set when DOWNLOAD_DATA is False"
        
        save_folder = os.path.join(local_output_path, self.relpath)
        remote_folder = os.path.join(cast(str, G.GS_PATH), self.relpath)
        
        # Enough training shards to cover the token budget.
        train_data_paths = [os.path.join(local_data_path, p) for p in get_train_files(self.pretrain_dataset, self.train_tokens)]
        
        # Pretrain (perplexity) validation sets; list-valued and dict-valued (masked) entries.
        val_info = LIST_OF_PRETRAIN_FILES[self.pretrain_dataset]["val"]
        eval_datasets = {}
        eval_datasets["pretrain"] = {
            k: [os.path.join(local_data_path, p) for p in v] if isinstance(v, list) else v
            for k, v in val_info.items()
        }

        for k, v in val_info.items():
            if isinstance(v, dict):
                eval_datasets["pretrain"][k] = {mk: [os.path.join(local_data_path, p) for p in mv] for mk, mv in v.items()}
        
        eval_datasets["sft"] = {}

        # dolma2 vocab/embedding sizing overrides.
        model_overrides = {'vocab_size': 100278, 'embedding_size': 100352, 'eos_token_id': 100257, 'pad_token_id': 100277}

        config = get_train_config(
            run_name=self.run_name,
            save_folder=save_folder,
            train_data_paths=train_data_paths,
            eval_datasets=eval_datasets,
            model_size=self.model_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            sam_rho=self.sam_rho,
            sam_base_optimizer=self.sam_base_optimizer,
            max_duration=f"{self.train_tokens}e9T",
            stop_at=None,
            seed=6198,
            model_overrides=model_overrides,
            scheduler_name=self.scheduler_name,
            scheduler_alpha_f=self.scheduler_alpha_f,
            scheduler_t_warmup=self.scheduler_t_warmup,
            global_train_batch_size=self.batch_size,
            device_train_microbatch_size=4,
            eval_interval=20000,
            save_interval_unsharded=5000,
            wandb_project=cast(str, G.PROJECT_NAME),
            wandb_entity=cast(str, G.WANDB_ENTITY),
            wandb_id=self.run_name,
            wandb_resume='allow',
            try_load_latest_save=True,
            run_sync_cmd=True,
            tokenizer={
                'identifier': 'tokenizers/allenai_dolma2.json',
                'truncate_direction': 'right',
            },
            dtype='uint32'
        )

        # Resume support: pull any partial results for this run, then write the config.
        builder.set_env("SYNC_CMD", f"gsutil -m rsync -r {save_folder}/ {remote_folder}/")
        builder.ensure_directory(save_folder)
        
        builder.rsync_from_gs(remote_folder, save_folder, delete=True, checksum=False, skip_existing=False, check_exists=True, contents=True)
        
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(config_path, config)

        # Optionally fetch the data shards from GCS to local disk before training.
        if G.DOWNLOAD_DATA:
            all_paths = train_data_paths.copy()
            for v in eval_datasets["pretrain"].values():
                all_paths.extend(v if isinstance(v, list) else v['data'] + v.get('masks', []))
            
            unique_dirs = {os.path.dirname(p) for p in all_paths}
            gs_data_path = cast(str, G.GS_DATA_PATH)
            for local_dir in unique_dirs:
                gs_dir = local_dir.replace(local_data_path, gs_data_path)
                builder.download_from_gs(gs_dir, local_dir, directory=True)

        # Launch distributed training, then sync the finished run back to GCS.
        olmo_path = cast(str, Project.config.OLMO_PATH)
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d '
            f'--nproc_per_node={self.pretrain_gpus} {train_script} {config_path}'
        )

        builder.rsync_to_gs(save_folder, remote_folder, delete=True, checksum=False, contents=True)


@dataclass(frozen=True)
class MidtrainedModel(Artifact):
    """
    MidtrainedModel for OLMo2-1B stage-2 (midtraining from a pre-trained checkpoint).
    Uses hyperparameters from configs/official-0425/OLMo2-1B-stage2-seed42.yaml.
    Data paths are loaded from launch/utils/midtrain_data.json.
    """
    optimizer: str = 'adamw'                     # 'adamw' or 'sam'
    sam_rho: float = 0.05                         # SAM neighborhood size (only when optimizer == 'sam')
    train_tokens: int = 4                         # pretraining budget of the base checkpoint (trillions)
    midtrain_gpus: int = 8
    global_train_batch_size: int = 1024
    midtrain_tokens: int = 5                      # midtraining budget, in billions of tokens
    seed: int = 42
    per_device_train_batch_size: int = 8
    sequence_length: int = 2048
    scheduler_alpha_f: float = 0.0

    @property
    def relpath(self) -> str:
        return f'MidtrainedModel/{self.run_name}'

    @property
    def run_name(self) -> str:
        name = f'OLMo2-1b-tk{self.train_tokens}T-adamw-Midtrain-{self.midtrain_tokens}B-{self.optimizer}'
        if self.optimizer == "sam":
            name += f'-rho{self.sam_rho:.0e}'.replace('e-0', 'e-')
        name += f'-bs{self.global_train_batch_size}'
        return name

    @property
    def checkpoint_relpath(self) -> str:
        return f'{self.relpath}/final-unsharded'
    
    @property
    def load_path(self) -> str:
        # Base 1B pretrained checkpoint that midtraining continues from.
        return 'PretrainedModel/OLMo2-1b-tk4T-adamw/final-unsharded'
    
    @property
    def learning_rate(self) -> float:
        # Final-LR of the base run, so midtraining picks up where pretraining left off.
        return 7.4487e-5
    
    @property
    def pretrain_dataset(self) -> str:
        return "dolmino"

    @property
    def model_size(self) -> str:
        return "1b"

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder='MidtrainedModel')
        found = any(f.startswith(remote_path) for f in remote_files)
        status = "✓ EXISTS" if found else "❌ NOT found"
        log.info(f"[MidtrainedModel] {status}: {self.run_name}")
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "gpus": self.midtrain_gpus,
            "nodes": 1,
            "cpus": self.midtrain_gpus * 2,
            "mem": '256GB',
            "requeue": True,
            "partition": 'flame',
            "qos": 'flame-32gpu_qos',
            "account": 'aditirag',
            "time": "5-00:00:00"
        }

    def construct(self, builder: Task):
        # Download the base 1B checkpoint, build the stage-2 config, train, sync to GCS.
        local_output_path = G.get_random_local_path()
        local_data_path = local_output_path if G.DOWNLOAD_DATA else cast(str, G.LOCAL_DATA_PATH)
        save_folder = os.path.join(local_output_path, self.relpath)
        gs_path = cast(str, G.GS_PATH)
        remote_folder = os.path.join(gs_path, self.relpath)
        olmo_path = cast(str, Project.config.OLMO_PATH)

        pre_ckpt_rel = self.load_path
        local_pre_path = os.path.join(local_output_path, pre_ckpt_rel)
        builder.rsync_from_gs(os.path.join(gs_path, pre_ckpt_rel), local_pre_path, delete=True, checksum=True, skip_existing=True, check_exists=True, contents=True)

        # Midtraining data mixture (paths defined in launch/utils/midtrain_data.json).
        dataset_info = json.load(open(os.path.join(olmo_path, 'launch', 'utils', 'midtrain_data.json')))
        train_data_paths = [os.path.join(local_data_path, p) for p in dataset_info['train_data_paths']]

        # Track downstream task accuracy during midtraining.
        eval_datasets = {'downstream': [
            'winogrande',
            'mmlu_other_var',
            'sciq',
            'hellaswag',
            'copa',
            'openbook_qa',
        ]}

        config = get_train_config(
            run_name=self.run_name,
            save_folder=save_folder,
            train_data_paths=train_data_paths,
            eval_datasets=eval_datasets,
            model_size='1b',
            optimizer=self.optimizer,
            sam_rho=self.sam_rho,
            learning_rate=self.learning_rate,
            weight_decay=0.1,
            optimizer_eps=1e-8,
            decay_embeddings=False,
            max_duration=f'{self.midtrain_tokens}e9T',
            seed=self.seed,
            scheduler_name='linear_with_warmup',
            scheduler_t_warmup=0,
            scheduler_alpha_f=self.scheduler_alpha_f,
            global_train_batch_size=self.global_train_batch_size,
            device_train_microbatch_size=self.per_device_train_batch_size,
            device_eval_batch_size=8,
            eval_interval=1000,
            save_interval_unsharded=4000,
            save_num_unsharded_checkpoints_to_keep=-1,
            load_path=local_pre_path,
            restore_dataloader=False,
            reset_optimizer_state=False,
            max_grad_norm=1.0,
            wandb_project=cast(str, G.PROJECT_NAME),
            wandb_entity=cast(str, G.WANDB_ENTITY),
            wandb_id=self.run_name,
            wandb_resume='allow',
            try_load_latest_save=True,
            run_sync_cmd=True,
            dtype='uint32',
            softmax_auxiliary_loss=True,
            auxiliary_loss_multiplier=1e-5,
            fused_loss=True,
            distributed_strategy='ddp',
            gen1_gc_interval=10,
        )
        config['tokenizer'] = {
            'identifier': 'tokenizers/allenai_dolma2.json',
            'truncate_direction': 'right',
        }
        config['data'].update({
            'num_workers': 8,
            'drop_last': True,
            'pin_memory': True,
            'prefetch_factor': 8,
            'persistent_workers': True,
            'timeout': 0,
            'pad_direction': 'right',
            'instance_filter': {
                'repetition_max_period': 13,
                'repetition_min_period': 1,
                'repetition_max_count': 32,
            },
        })
        builder.set_env("SYNC_CMD", f"gsutil -m rsync -r {save_folder}/ {remote_folder}/")
        builder.ensure_directory(save_folder)
        builder.rsync_from_gs(remote_folder, save_folder, delete=True, checksum=False, skip_existing=False, check_exists=True, contents=True)

        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(config_path, config)

        if G.DOWNLOAD_DATA:
            all_paths = train_data_paths.copy()
            unique_dirs = {os.path.dirname(p) for p in all_paths}
            gs_data_path = cast(str, G.GS_DATA_PATH)
            for local_dir in unique_dirs:
                gs_dir = local_dir.replace(local_data_path, gs_data_path)
                builder.download_from_gs(gs_dir, local_dir, directory=True)

        
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d '
            f'--nproc_per_node={self.midtrain_gpus} {train_script} {config_path}'
        )

        builder.rsync_to_gs(save_folder, remote_folder, delete=True, checksum=False, contents=True)



@dataclass(frozen=True)
class AnnealedModel(Artifact):
    """
    AnnealedModel loads a checkpoint from a PretrainedModel and continues training
    with the WSD (Warmup-Stable-Decay) scheduler.
    """
    pretrained_model: PretrainedModel            # base run whose mid-training checkpoint we anneal
    pt_token: int                                # token budget (B) identifying the checkpoint to anneal
    anneal_gpus: int = 8
    anneal_match: str = "token"                  # budget matching: "token" or "compute"
    anneal_optim: str = 'adamw'
    anneal_percent: int = 10                      # anneal length as a % of the checkpoint's step count

    @property
    def sequence_length(self) -> int:
        return self.pretrained_model.sequence_length

    @property
    def relpath(self) -> str:
        return f'AnnealedModel/{self.run_name}'

    @property
    def pretrain_ckpt_step(self) -> int:
        return ANNEAL_CKPT[self.anneal_percent][self.anneal_match][int(self.model_size[:-1])][self.pt_token]

    @property
    def train_tokens(self) -> int:
        return self.pretrained_model.train_tokens

    @property
    def checkpoint_relpath(self) -> str:
        return f'{self.relpath}/final-unsharded'

    @property
    def run_name(self) -> str:
        return f'{self.pretrained_model.run_name}-anneal-{self.anneal_optim}-ckpt{self.pretrain_ckpt_step}-percent{self.anneal_percent}'
    
    @property
    def pretrain_dataset(self) -> str:
        return self.pretrained_model.pretrain_dataset
    
    @property
    def model_size(self) -> str:
        return self.pretrained_model.model_size

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
            
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder='AnnealedModel')
        return any(f.startswith(remote_path) for f in remote_files)

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "gres": f"gpu:{self.anneal_gpus}",
            "nodes": 1,
            "cpus": self.anneal_gpus * 2,
            "mem": '64GB',
            "requeue": True,
            "partition": 'flame',
            "qos": 'flame-16gpu-c_qos',
            "account": 'aditirag',
            "time": "12-00:00:00"
        }

    def construct(self, builder: Task):

        assert self.anneal_percent is not None, "anneal_percent must be set"
        local_output_path = G.get_random_local_path()
        local_data_path = local_output_path if G.DOWNLOAD_DATA else G.LOCAL_DATA_PATH
        assert local_data_path is not None, "LOCAL_DATA_PATH must be set when DOWNLOAD_DATA is False"
        
        save_folder = os.path.join(local_output_path, self.relpath)
        gs_path = cast(str, G.GS_PATH)
        remote_folder = os.path.join(gs_path, self.relpath)

        ckpt_rel = f'{self.pretrained_model.relpath}/step{self.pretrain_ckpt_step}-unsharded'
        pretrained_model_gs_path = os.path.join(gs_path, ckpt_rel)
        local_checkpoint_path = os.path.join(local_output_path, ckpt_rel)

        log.info(f"Downloading checkpoint from: {pretrained_model_gs_path}")
        builder.rsync_from_gs(pretrained_model_gs_path, local_checkpoint_path, delete=True, checksum=False, skip_existing=False, check_exists=True, contents=True)

        # Anneal for `anneal_percent`% of the checkpoint's steps; derive matching token budget
        # (batch=256 * seq_len) so we pull enough decay-stage shards. max_duration is a step count.
        anneal_tokens = (self.anneal_percent * self.pretrain_ckpt_step / 100) * self.sequence_length * 256 / BILLION
        max_duration = int(self.anneal_percent * self.pretrain_ckpt_step / 100)

        train_data_paths = [
            os.path.join(local_data_path, p) 
            for p in get_train_files(self.pretrain_dataset, int(anneal_tokens) + 1, train_stage='decay')
        ]

        val_info = LIST_OF_PRETRAIN_FILES[self.pretrain_dataset]["val"]
        eval_datasets = {}
        eval_datasets["pretrain"] = {}
        eval_datasets["sft"] = {}
        for k, v in val_info.items():
            if isinstance(v, list):
                eval_datasets["pretrain"][k] = [os.path.join(local_data_path, p) for p in v]
            elif isinstance(v, dict):
                eval_datasets["pretrain"][k] = {mk: [os.path.join(local_data_path, p) for p in mv] for mk, mv in v.items()}


        model_overrides = {'vocab_size': 100278, 'embedding_size': 100352, 'eos_token_id': 100257, 'pad_token_id': 100277}

        config = get_train_config(
            run_name=self.run_name,
            save_folder=save_folder,
            train_data_paths=train_data_paths,
            eval_datasets=eval_datasets,
            model_size=self.model_size,
            optimizer=self.anneal_optim,
            learning_rate=self.pretrained_model.learning_rate,
            weight_decay=self.pretrained_model.weight_decay,
            sam_rho=self.pretrained_model.sam_rho,
            sam_base_optimizer=self.pretrained_model.sam_base_optimizer,
            max_duration=max_duration,
            stop_at=None,
            seed=6198,
            model_overrides=model_overrides,
            scheduler_name='linear_with_warmup',
            scheduler_t_warmup=0,
            global_train_batch_size=self.pretrained_model.batch_size,
            device_train_microbatch_size=4,
            eval_interval=5000,
            save_interval_unsharded=5000,
            wandb_project=cast(str, G.PROJECT_NAME),
            wandb_entity=cast(str, G.WANDB_ENTITY),
            wandb_id=self.run_name,
            wandb_resume='allow',
            load_path=local_checkpoint_path,
            reset_optimizer_state=False,
            restore_dataloader=False,
            try_load_latest_save=True,
            run_sync_cmd=True,
            tokenizer={
                'identifier': 'tokenizers/allenai_dolma2.json',
                'truncate_direction': 'right',
            },
            dtype='uint32',
        )

        builder.set_env("SYNC_CMD", f"gsutil -m rsync -r {save_folder}/ {remote_folder}/")
        builder.ensure_directory(save_folder)
        
        builder.rsync_from_gs(remote_folder, save_folder, delete=True, checksum=False, skip_existing=False, check_exists=True, contents=True)
        
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(config_path, config)

        if G.DOWNLOAD_DATA:
            all_paths = train_data_paths.copy()
            for v in eval_datasets["pretrain"].values():
                all_paths.extend(v if isinstance(v, list) else v['data'] + v.get('masks', []))
            
            unique_dirs = {os.path.dirname(p) for p in all_paths}
            gs_data_path = cast(str, G.GS_DATA_PATH)
            for local_dir in unique_dirs:
                gs_dir = local_dir.replace(local_data_path, gs_data_path)
                builder.download_from_gs(gs_dir, local_dir, directory=True)

        olmo_path = cast(str, Project.config.OLMO_PATH)
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d '
            f'--nproc_per_node={self.anneal_gpus} {train_script} {config_path}'
        )

        builder.rsync_to_gs(save_folder, remote_folder, delete=True, checksum=False, contents=True)


@dataclass(frozen=True)
class AnnealedModel2(Artifact):
    """Anneal variant driven by the WSD scheduler's built-in decay.

    Unlike ``AnnealedModel`` (which trains a separate decay run), this loads a base
    checkpoint and runs the full ``wsd`` schedule with ``anneal_begin`` set from
    ``ANNEAL_CKPT2``, so the LR decays in-place over the remaining steps.
    """
    pretrained_model: PretrainedModel
    pt_token: int                                # token budget (B); sets WSD max_duration
    anneal_gpus: int = 8
    anneal_optim: str = 'adamw'
    anneal_percent: int = 10                      # selects the anneal_begin step from ANNEAL_CKPT2

    @property
    def sequence_length(self) -> int:
        return self.pretrained_model.sequence_length

    @property
    def relpath(self) -> str:
        return f'AnnealedModel2/{self.run_name}'

    @property
    def pretrain_ckpt_step(self) -> int:
        return ANNEAL_CKPT2[int(self.model_size[:-1])][self.anneal_percent]

    @property
    def train_tokens(self) -> int:
        return self.pretrained_model.train_tokens

    @property
    def checkpoint_relpath(self) -> str:
        return f'{self.relpath}/final-unsharded'

    @property
    def run_name(self) -> str:
        return f'{self.pretrained_model.run_name}-anneal-{self.anneal_optim}-ckpt{self.pretrain_ckpt_step}-percent{self.anneal_percent}'
    
    @property
    def pretrain_dataset(self) -> str:
        return self.pretrained_model.pretrain_dataset
    
    @property
    def model_size(self) -> str:
        return self.pretrained_model.model_size

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
            
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder='AnnealedModel2')
        return any(f.startswith(remote_path) for f in remote_files)

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "gpus": self.anneal_gpus,
            "nodes": 1,
            "cpus": self.anneal_gpus * 2,
            "mem": '64GB',
            "requeue": True,
            'partition': 'flame',
            'account': 'aditirag',
            'qos': 'flame-32gpu_qos',
            'time': '4-00:00:00',
        }

    def construct(self, builder: Task):
        local_output_path = G.get_random_local_path()
        local_data_path = local_output_path if G.DOWNLOAD_DATA else G.LOCAL_DATA_PATH
        assert local_data_path is not None, "LOCAL_DATA_PATH must be set when DOWNLOAD_DATA is False"
        
        save_folder = os.path.join(local_output_path, self.relpath)
        gs_path = cast(str, G.GS_PATH)
        remote_folder = os.path.join(gs_path, self.relpath)

        ckpt_rel = f'{self.pretrained_model.relpath}/step{self.pretrain_ckpt_step}-unsharded'
        pretrained_model_gs_path = os.path.join(gs_path, ckpt_rel)
        local_checkpoint_path = os.path.join(local_output_path, ckpt_rel)

        log.info(f"Downloading checkpoint from: {pretrained_model_gs_path}")
        builder.rsync_from_gs(pretrained_model_gs_path, local_checkpoint_path, delete=True, checksum=False, skip_existing=False, check_exists=True, contents=True)

        train_data_paths = [os.path.join(local_data_path, p) for p in get_train_files(self.pretrain_dataset, self.train_tokens)]

        val_info = LIST_OF_PRETRAIN_FILES[self.pretrain_dataset]["val"]
        eval_datasets = {}
        eval_datasets["pretrain"] = {}
        eval_datasets["sft"] = {}
        for k, v in val_info.items():
            if isinstance(v, list):
                eval_datasets["pretrain"][k] = [os.path.join(local_data_path, p) for p in v]
            elif isinstance(v, dict):
                eval_datasets["pretrain"][k] = {mk: [os.path.join(local_data_path, p) for p in mv] for mk, mv in v.items()}

        model_overrides = {'vocab_size': 100278, 'embedding_size': 100352, 'eos_token_id': 100257, 'pad_token_id': 100277}

        config = get_train_config(
            run_name=self.run_name,
            save_folder=save_folder,
            train_data_paths=train_data_paths,
            eval_datasets=eval_datasets,
            model_size=self.model_size,
            optimizer=self.anneal_optim,
            learning_rate=self.pretrained_model.learning_rate,
            weight_decay=self.pretrained_model.weight_decay,
            sam_rho=self.pretrained_model.sam_rho,
            sam_base_optimizer=self.pretrained_model.sam_base_optimizer,
            max_duration=f"{self.pt_token}e9T",
            stop_at=None,
            seed=6198,
            model_overrides=model_overrides,
            scheduler_name='wsd',
            scheduler_t_warmup=WARMUP_STEPS[int(self.model_size[:-1])],
            scheduler_anneal_begin=self.pretrain_ckpt_step,
            global_train_batch_size=self.pretrained_model.batch_size,
            device_train_microbatch_size=16,
            eval_interval=20000,
            save_interval_unsharded=5000,
            wandb_project=cast(str, G.PROJECT_NAME),
            wandb_entity=cast(str, G.WANDB_ENTITY),
            wandb_id=self.run_name,
            wandb_resume='allow',
            load_path=local_checkpoint_path,
            reset_optimizer_state=False,
            restore_dataloader=True,
            try_load_latest_save=True,
            run_sync_cmd=True,
            tokenizer={
                'identifier': 'tokenizers/allenai_dolma2.json',
                'truncate_direction': 'right',
            },
            dtype='uint32',
        )

        builder.set_env("SYNC_CMD", f"gsutil -m rsync -r {save_folder}/ {remote_folder}/")
        builder.ensure_directory(save_folder)
        
        builder.rsync_from_gs(remote_folder, save_folder, delete=True, checksum=False, skip_existing=False, check_exists=True, contents=True)
        
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(config_path, config)

        if G.DOWNLOAD_DATA:
            all_paths = train_data_paths.copy()
            for v in eval_datasets["pretrain"].values():
                all_paths.extend(v if isinstance(v, list) else v['data'] + v.get('masks', []))
            
            unique_dirs = {os.path.dirname(p) for p in all_paths}
            gs_data_path = cast(str, G.GS_DATA_PATH)
            for local_dir in unique_dirs:
                gs_dir = local_dir.replace(local_data_path, gs_data_path)
                builder.download_from_gs(gs_dir, local_dir, directory=True)

        olmo_path = cast(str, Project.config.OLMO_PATH)
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d '
            f'--nproc_per_node={self.anneal_gpus} {train_script} {config_path}'
        )

        builder.rsync_to_gs(save_folder, remote_folder, delete=True, checksum=False, contents=True)


@dataclass(frozen=True)
class SFTModel(Artifact):
    """Supervised fine-tuning of any upstream checkpoint on an SFT dataset.

    Loads ``pretrained_model``'s final checkpoint and fine-tunes (with label masking)
    on ``sft_dataset``, also tracking pretrain perplexity to measure forgetting.
    """
    train_tokens: int                            # SFT budget, in millions of tokens
    pretrained_model: PretrainedModel | AnnealedModel | AnnealedModel2 | MidtrainedModel
    sft_dataset: str                             # key into LIST_OF_SFT_FILES
    optimizer: str = 'adamw'
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    batch_size: int = 64
    scheduler_name: str = 'cosine_with_warmup'
    scheduler_alpha_f: float = 0.1
    sft_gpus: int = 1
    step: str = 'final'                           # which upstream checkpoint step to load

    @property
    def sequence_length(self) -> int:
        return self.pretrained_model.sequence_length

    @property
    def relpath(self) -> str:
        return f'SFTModel/{self.sft_dataset}/{self.run_name}'

    @property
    def checkpoint_relpath(self) -> str:
        return f'{self.relpath}/final-unsharded'

    @property
    def pretrain_dataset(self) -> str:
        return self.pretrained_model.pretrain_dataset

    @property
    def run_name(self) -> str:
        pre_name = self.pretrained_model.run_name
        lr_str = f'{self.learning_rate:.2e}'.replace('e-0', 'e-')
        wd_str = f'{self.weight_decay:.0e}'.replace('e-0', 'e-') if self.weight_decay > 0 else '0'
        
        name = f'{pre_name}-SFT-{self.sft_dataset}-tk{self.train_tokens}M-lr{lr_str}-wd{wd_str}-bs{self.batch_size}'
        return name

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder=f'SFTModel/{self.sft_dataset}')
        found = any(f.startswith(remote_path) for f in remote_files)
        log.info(f"[SFTModel] {'✓ EXISTS' if found else '❌ NOT found'}: {self.sft_dataset}")
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {
            'gpus': f"A100_40GB:{str(self.sft_gpus)}",
            'nodes': 1,
            'cpus': self.sft_gpus * 2,
            'mem': '64GB',
            'requeue': True,
            "partition": 'general',
            "time": "2-00:00:00"
        }

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        save_folder = os.path.join(local_root, self.relpath)
        gs_root = cast(str, Project.config.GS_PATH)
        remote_folder = os.path.join(gs_root, self.relpath)

        # Upstream checkpoint to fine-tune from.
        pre_ckpt_rel = self.pretrained_model.checkpoint_relpath
        local_pre_path = os.path.join(local_root, pre_ckpt_rel)
        builder.rsync_from_gs(os.path.join(gs_root, pre_ckpt_rel), local_pre_path, delete=True, checksum=True, skip_existing=True, check_exists=True, contents=True, )

        # SFT train inputs + label masks (only completion tokens contribute to loss).
        dataset_info = LIST_OF_SFT_FILES[self.pretrain_dataset][self.sft_dataset]
        train_paths = [os.path.join(local_root, p) for p in dataset_info["data_paths"]]
        mask_paths = [os.path.join(local_root, p) for p in dataset_info["mask_paths"]]
        tmp_train_tokens = dataset_info.get("train_tokens", None)

        # Three eval buckets: pretrain ppl (forgetting), SFT ppl, and downstream tasks.
        eval_datasets = {}
        eval_datasets["pretrain"] = {}
        eval_datasets["sft"] = {}
        eval_datasets["downstream"] = [
            'winogrande',
            'mmlu_other_var',
            'sciq',
            'hellaswag',
            'copa',
            'openbook_qa',
        ]
        for k, v in dataset_info["val"].items():
            paths = v.get("data", []) if isinstance(v, dict) else v
            masks = v.get("masks", None) if isinstance(v, dict) else None
            if masks is None:
                eval_datasets["pretrain"][k] = [os.path.join(local_root, p) for p in paths]
            else:
                eval_datasets["sft"][k] = {
                    "data_paths" : [os.path.join(local_root, p) for p in paths],
                    "mask_paths" : [os.path.join(local_root, m) for m in masks]
                }
        
        if self.pretrain_dataset == "dolmino":
            del eval_datasets["pretrain"]


        # A dataset may pin its own token budget; otherwise use the requested one.
        train_tokens = self.train_tokens
        if tmp_train_tokens is not None:
            log.info(f"Using train tokens from dataset info: {train_tokens}M")
            train_tokens = tmp_train_tokens

        # Derive total/warmup steps (10% warmup) from the token budget.
        total_tokens = train_tokens * MILLION
        total_steps = max(1, total_tokens // (self.batch_size * self.sequence_length))
        warmup_steps = max(1, int(total_steps * 0.1))

        overrides = {'vocab_size': 100278, 'embedding_size': 100352, 'eos_token_id': 100257, 'pad_token_id': 100277}

        # wandb run id is capped at 64 chars; keep the most-specific suffix of the run name.
        exp_name = self.run_name[len(self.run_name)-64+1:] if len(self.run_name) > 64 else self.run_name

        config = get_train_config(
            run_name=self.run_name,
            save_folder=save_folder,
            train_data_paths=train_paths,
            train_data_label_mask_paths=mask_paths,
            model_size=self.pretrained_model.model_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_duration=f'{train_tokens}e6T',
            decay_embeddings=False if self.pretrained_model.model_size == "1b" else True,
            seed=6198,
            model_overrides=overrides,
            scheduler_name=self.scheduler_name,
            scheduler_alpha_f=self.scheduler_alpha_f,
            scheduler_t_warmup=warmup_steps,
            global_train_batch_size=self.batch_size,
            device_train_microbatch_size=4,
            eval_interval=10000,
            save_interval_unsharded=1000,
            wandb_project=cast(str, Project.config.PROJECT_NAME),
            wandb_entity=cast(str, Project.config.WANDB_ENTITY),
            wandb_id=exp_name,
            load_path=local_pre_path,
            reset_optimizer_state=True,
            tokenizer={'identifier': 'tokenizers/allenai_dolma2.json'},
            dtype='uint32',
        )

        builder.ensure_directory(save_folder)
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(os.path.join(save_folder, 'config.yaml'), config)

        gs_data_path = cast(str, Project.config.GS_DATA_PATH)            
        for p in (train_paths + mask_paths + [item for sub_dict in eval_datasets["sft"].values() for sub_list in sub_dict.values() for item in sub_list]):
            builder.download_from_gs(p.replace(local_root, gs_data_path), p, directory=False)
        if self.pretrain_dataset != "dolmino":
            for p in [item for sub in eval_datasets["pretrain"].values() for item in sub]:
                builder.download_from_gs(p.replace(local_root, gs_data_path), p, directory=False)

        olmo_path = cast(str, Project.config.OLMO_PATH)
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && '
            f'torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d --nproc_per_node={self.sft_gpus} {train_script} {config_path}'
        )
        builder.upload_to_gs(save_folder, remote_folder, directory=True)

@dataclass(frozen=True)
class EWC_SFT(Artifact):
    """SFT finetuning with Elastic Weight Consolidation (``scripts/train_ewc.py``).

    Same inputs and training layout as ``SFTModel``, but checkpoints live under ``EWC_SFT/`` and
    Fisher uses a subsample of pretrain train memmaps (``ewc_fisher_pretrain_subsample_tokens_billion``).
    """

    train_tokens: int
    pretrained_model: PretrainedModel | AnnealedModel | AnnealedModel2 | MidtrainedModel
    sft_dataset: str
    optimizer: str = 'adamw'
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    batch_size: int = 64
    scheduler_name: str = 'cosine_with_warmup'
    scheduler_alpha_f: float = 0.1
    sft_gpus: int = 1
    step: str = 'final'
    ewc_lambda: float = 4000.0                    # EWC penalty strength on drift from pretrain weights
    ewc_fisher_batches: int = 100                 # batches used to estimate the Fisher diagonal
    ewc_fisher_pretrain_subsample_tokens_billion: float = 1.0  # pretrain tokens sampled for Fisher

    @property
    def sequence_length(self) -> int:
        return self.pretrained_model.sequence_length

    @property
    def relpath(self) -> str:
        return f'EWC_SFT/{self.sft_dataset}/{self.run_name}'

    @property
    def checkpoint_relpath(self) -> str:
        return f'{self.relpath}/final-unsharded'

    @property
    def pretrain_dataset(self) -> str:
        return self.pretrained_model.pretrain_dataset

    @property
    def run_name(self) -> str:
        pre_name = self.pretrained_model.run_name
        lr_str = f'{self.learning_rate:.2e}'.replace('e-0', 'e-')
        wd_str = f'{self.weight_decay:.0e}'.replace('e-0', 'e-') if self.weight_decay > 0 else '0'
        base = f'{pre_name}-EWC_SFT-{self.sft_dataset}-tk{self.train_tokens}M-lr{lr_str}-wd{wd_str}-bs{self.batch_size}'
        lam = f'{self.ewc_lambda:.2e}'.replace('e-0', 'e-')
        pt = f'{self.ewc_fisher_pretrain_subsample_tokens_billion:.5g}'.replace('.', 'p')
        return f'{base}-lam{lam}-fb{self.ewc_fisher_batches}-fisher{pt}B'

    def get_requirements(self) -> Dict[str, Any]:
        return {
            'gpus': f"A100_40GB:{str(self.sft_gpus)}",
            'nodes': 1,
            'cpus': self.sft_gpus * 2,
            'mem': '64GB',
            'requeue': True,
            "partition": 'general',
            "time": "2-00:00:00"
        }

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder=f'EWC_SFT/{self.sft_dataset}')
        found = any(f.startswith(remote_path) for f in remote_files)
        log.info(f"[EWC_SFT] {'✓ EXISTS' if found else '❌ NOT found'}: {self.sft_dataset}")
        return found

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        save_folder = os.path.join(local_root, self.relpath)
        gs_root = cast(str, Project.config.GS_PATH)
        remote_folder = os.path.join(gs_root, self.relpath)

        if self.step != 'final':
            pre_ckpt_rel = f"{self.pretrained_model.relpath}/{self.step}-unsharded"
        else:
            pre_ckpt_rel = self.pretrained_model.checkpoint_relpath
        local_pre_path = os.path.join(local_root, pre_ckpt_rel)
        builder.rsync_from_gs(os.path.join(gs_root, pre_ckpt_rel), local_pre_path, delete=True, checksum=True, skip_existing=True, check_exists=True, contents=True, )

        dataset_info = LIST_OF_SFT_FILES[self.pretrain_dataset][self.sft_dataset]
        train_paths = [os.path.join(local_root, p) for p in dataset_info["data_paths"]]
        mask_paths = [os.path.join(local_root, p) for p in dataset_info["mask_paths"]]
        tmp_train_tokens = dataset_info.get("train_tokens", None)

        eval_datasets = {}
        eval_datasets["pretrain"] = {}
        eval_datasets["sft"] = {}
        for k, v in dataset_info["val"].items():
            paths = v.get("data", []) if isinstance(v, dict) else v
            masks = v.get("masks", None) if isinstance(v, dict) else None
            if masks is None:
                eval_datasets["pretrain"][k] = [os.path.join(local_root, p) for p in paths]
            else:
                eval_datasets["sft"][k] = {
                    "data_paths" : [os.path.join(local_root, p) for p in paths],
                    "mask_paths" : [os.path.join(local_root, m) for m in masks]
                }

        if self.pretrain_dataset == "dolmino":
            del eval_datasets["pretrain"]

        ewc_fisher_paths = pretrain_train_memmap_paths_for_ewc_fisher_subsample(
            self.pretrain_dataset,
            local_root,
            self.ewc_fisher_pretrain_subsample_tokens_billion,
        )

        train_tokens = self.train_tokens
        if tmp_train_tokens is not None:
            log.info(f"Using train tokens from dataset info: {train_tokens}M")
            train_tokens = tmp_train_tokens

        total_tokens = train_tokens * MILLION
        total_steps = max(1, total_tokens // (self.batch_size * self.sequence_length))
        warmup_steps = max(1, int(total_steps * 0.1))

        overrides = {'vocab_size': 100278, 'embedding_size': 100352, 'eos_token_id': 100257, 'pad_token_id': 100277}

        exp_name = self.run_name[len(self.run_name)-64+1:] if len(self.run_name) > 64 else self.run_name

        config = get_train_config(
            run_name=self.run_name,
            save_folder=save_folder,
            train_data_paths=train_paths,
            train_data_label_mask_paths=mask_paths,
            model_size=self.pretrained_model.model_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_duration=f'{train_tokens}e6T',
            eval_datasets=eval_datasets,
            decay_embeddings=False if self.pretrained_model.model_size == "1b" else True,
            seed=6198,
            model_overrides=overrides,
            scheduler_name=self.scheduler_name,
            scheduler_alpha_f=self.scheduler_alpha_f,
            scheduler_t_warmup=warmup_steps,
            global_train_batch_size=self.batch_size,
            device_train_microbatch_size=4,
            eval_interval=10000,
            save_interval_unsharded=1000,
            wandb_project=cast(str, Project.config.PROJECT_NAME),
            wandb_entity=cast(str, Project.config.WANDB_ENTITY),
            wandb_id=exp_name,
            load_path=local_pre_path,
            reset_optimizer_state=True,
            tokenizer={'identifier': 'tokenizers/allenai_dolma2.json'},
            dtype='uint32',
            ewc_lambda=self.ewc_lambda,
            ewc_fisher_batches=self.ewc_fisher_batches,
            ewc_fisher_paths=ewc_fisher_paths,
        )

        builder.ensure_directory(save_folder)
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(os.path.join(save_folder, 'config.yaml'), config)

        gs_data_path = cast(str, Project.config.GS_DATA_PATH)
        for p in (train_paths + mask_paths + [item for sub_dict in eval_datasets["sft"].values() for sub_list in sub_dict.values() for item in sub_list]):
            builder.download_from_gs(p.replace(local_root, gs_data_path), p, directory=False)
        if self.pretrain_dataset != "dolmino":
            for p in [item for sub in eval_datasets["pretrain"].values() for item in sub]:
                builder.download_from_gs(p.replace(local_root, gs_data_path), p, directory=False)
        for p in ewc_fisher_paths:
            builder.download_from_gs(p.replace(local_root, gs_data_path), p, directory=False)

        olmo_path = cast(str, Project.config.OLMO_PATH)
        train_script = os.path.join(olmo_path, 'scripts', 'train_ewc.py')
        builder.run_command(
            f'cd {olmo_path} && '
            f'torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d --nproc_per_node={self.sft_gpus} {train_script} {config_path}'
        )
        builder.upload_to_gs(save_folder, remote_folder, directory=True)


@dataclass(frozen=True)
class PerturbedModel(Artifact):
    """Add Gaussian noise (std=``sigma``) to a checkpoint's weights.

    Runs ``new_utils/perturb_weights.py`` directly on the GCS checkpoint to probe
    loss-landscape sharpness; no training involved.
    """
    base_model: "PretrainedModel | AnnealedModel | MidtrainedModel | AnnealedModel2"
    sigma: float                                 # std of the additive Gaussian noise
    seed: int = 64
    device: str = "cpu"

    @staticmethod
    def _format_sigma(sigma: float) -> str:
        # Filesystem-safe rendering of sigma for the run name.
        return f"{sigma:.2e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")

    @property
    def run_name(self) -> str:
        return f"{self.base_model.run_name}_perturbed_{self._format_sigma(self.sigma)}"

    @property
    def relpath(self) -> str:
        return f"PerturbedModel/{self.run_name}"

    @property
    def checkpoint_relpath(self) -> str:
        return f"{self.relpath}/final-unsharded"

    @property
    def pretrain_dataset(self) -> str:
        return self.base_model.pretrain_dataset

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder="PerturbedModel")
        found = any(f.startswith(remote_path) for f in remote_files)
        log.info(f"[PerturbedModel] {'✓ EXISTS' if found else '❌ NOT found'}: {self.run_name}")
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "gres": f"gpu:1",
            "nodes": 1,
            "cpus-per-task": 4,
            "mem": "256GB",
            "requeue": True,
            "partition": "general",
            "time": "1-00:00:00",
        }

    def construct(self, builder: Task):
        gs_root = cast(str, Project.config.GS_PATH)
        if isinstance(self.base_model, PretrainedModel):
            base_subfolder = "PretrainedModel"
        elif isinstance(self.base_model, AnnealedModel) or isinstance(self.base_model, AnnealedModel2):
            base_subfolder = "AnnealedModel"
        elif isinstance(self.base_model, MidtrainedModel):
            base_subfolder = "MidtrainedModel"
        else:
            raise TypeError(f"Unsupported base model type: {type(self.base_model)}")

        gcs_dir = os.path.join(gs_root, base_subfolder)
        output_gcs_dir = os.path.join(gs_root, "PerturbedModel")

        olmo_path = cast(str, Project.config.OLMO_PATH)
        perturb_script = os.path.join(olmo_path, "new_utils", "perturb_weights.py")
        cmd_parts = [
            "cd", olmo_path, "&&",
            "python", perturb_script,
            f"--gcs_dir {gcs_dir}",
            f"--model_name {self.base_model.run_name}",
            f"--output_gcs_dir {output_gcs_dir}",
            f"--sigma {self.sigma}",
            f"--seed {self.seed}",
            f"--device {self.device}",
        ]
        builder.run_command(" ".join(cmd_parts))


@dataclass(frozen=True)
class ModelEvaluation(Artifact):
    """Perplexity/loss evaluation of any model on its pretrain + SFT validation sets.

    Picks the right val sets for the model type, runs ``scripts/evaluate.py``, and
    writes a per-model JSON to GCS. Optionally evaluates a quantized HF checkpoint.
    """
    model: "PretrainedModel | SFTModel | EWC_SFT | AnnealedModel | PerturbedModel | MidtrainedModel | AnnealedModel2"
    device: str = 'cuda'
    chunk_size: int = 1024
    hf_model: bool = False                        # evaluate an HF-converted checkpoint
    quant_bit: int = 4                            # quantization bit-width (HF only)

    @property
    def relpath(self) -> str:
        if self.quant_bit is not None:
            assert self.hf_model
            return f'ModelEvaluation/{self.model.run_name}-quant-{self.quant_bit}bit-eval.json'
        return f'ModelEvaluation/{self.model.run_name}-eval.json'

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.relpath)
        remote_files = G.get_remote_files(subfolder='ModelEvaluation')
        found = any(f.startswith(remote_path) for f in remote_files)
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {'gres': 'gpu:1', 'nodes': 1, 'cpus': 4, 'mem': '64GB', 'partition': 'general', 'time': '1-00:00:00'}

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        local_output = os.path.join(local_root, self.relpath)
        builder.ensure_directory(os.path.dirname(local_output))

        local_model = os.path.join(local_root, self.model.checkpoint_relpath)
        builder.rsync_from_gs(os.path.join(cast(str, Project.config.GS_PATH), self.model.checkpoint_relpath), local_model, delete=False, checksum=False, skip_existing=False, check_exists=True, contents=True)

        eval_meta = (LIST_OF_SFT_FILES[self.model.pretrain_dataset][self.model.sft_dataset]["val"]
                     if isinstance(self.model, (SFTModel, EWC_SFT)) else
                     LIST_OF_PRETRAIN_FILES[self.model.pretrain_dataset]["val"])
        
        target_evals, target_masks = [], []
        gs_data_path = cast(str, Project.config.GS_DATA_PATH)

        for meta in eval_meta.values():
            paths = meta.get("data", []) if isinstance(meta, dict) else meta
            masks = meta.get("masks", []) if isinstance(meta, dict) else []
            
            for i, p in enumerate(paths):
                lp = os.path.join(local_root, p)
                builder.download_from_gs(lp.replace(local_root, gs_data_path), lp, directory=False)
                target_evals.append(lp)
                
                if i < len(masks):
                    lm = os.path.join(local_root, masks[i])
                    builder.download_from_gs(lm.replace(local_root, gs_data_path), lm, directory=False)
                    target_masks.append(lm)
                else:
                    target_masks.append('')

        olmo_path = cast(str, Project.config.OLMO_PATH)
        eval_script = os.path.join(olmo_path, 'scripts', 'evaluate.py')
        cmd = [
            'python', eval_script,
            f"--model_path {local_model}", f"--device {self.device}",
            f"--data_path {','.join(target_evals)}", f"--chunk_size {self.chunk_size}",
            f"--output_path {local_output}"
        ]
        if any(target_masks):
            cmd.append(f"--mask_path {','.join(target_masks)}")
        cmd.append(f"--dtype=uint32")
        if self.hf_model:
            cmd.append("--hf_model")
            if self.quant_bit is not None:
                cmd.append(f"--quantize={self.quant_bit}")
        
        builder.run_command(' '.join(cmd))
        builder.rsync_to_gs(os.path.dirname(local_output), os.path.join(cast(str, Project.config.GS_PATH), 'ModelEvaluation'), delete=False, checksum=False, contents=True)


@dataclass(frozen=True)
class ModelEvaluationDownstream(Artifact):
    """
    Evaluate an HFModel on downstream tasks using the olmes evaluation framework.
    """
    model: "HFModel"
    tasks: tuple = ('core_9mcqa::olmes', 'mmlu:mc::olmes', 'olmo_2_generative::olmes', 'olmo_2_heldout::olmes')
    batch_size: int = 8
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    gpu_count: int = 1
    olmes_chat_template: str = 'tokenize_data'

    @property
    def run_name(self) -> str:
        name = self.model.run_name
        if self.load_in_4bit:
            name += '-4bit'
        elif self.load_in_8bit:
            name += '-8bit'
        return name

    @property
    def relpath(self) -> str:
        return f'ModelEvaluationDownstream/{self.run_name}-downstream-eval.jsonl'

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.relpath)
        remote_files = G.get_remote_files(subfolder='ModelEvaluationDownstream')
        found = any(f.startswith(remote_path) for f in remote_files)
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {
            'gpus': 1,
            'nodes': 1,
            'cpus-per-task': 4,
            'mem': '128GB',
            'partition': 'general',
            'time': '1-00:00:00',
        }

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        gs_root = cast(str, Project.config.GS_PATH)
        local_output = os.path.join(local_root, self.relpath)
        builder.ensure_directory(os.path.dirname(local_output))

        src_ckpt_rel = self.model.checkpoint_relpath
        local_model_path = os.path.join(local_root, src_ckpt_rel)
        builder.rsync_from_gs(
            os.path.join(gs_root, src_ckpt_rel),
            local_model_path,
            delete=False,
            checksum=False,
            skip_existing=False,
            check_exists=True,
            contents=True,
        )

        olmes_output_dir = os.path.join(local_root, f'{self.run_name}_output')
        builder.ensure_directory(olmes_output_dir)

        model_args_parts = [
            f'model_path={local_model_path}',
            f'chat_template={self.olmes_chat_template}',
        ]
        if self.load_in_4bit:
            model_args_parts.append('load_in_4bit=true')
        if self.load_in_8bit:
            model_args_parts.append('load_in_8bit=true')

        model_args_str = ','.join(model_args_parts)
        tasks_str = ' '.join(self.tasks)

        cmd_parts = [
            'source ~/miniconda3/etc/profile.d/conda.sh && conda activate olmes &&',
            'olmes',
            f'--model olmo-2-1b-0425-iwatts',
            f'--model-args "{model_args_str}"',
            f'--task {tasks_str}',
            f'--output-dir {olmes_output_dir}',
            f'--batch-size {self.batch_size}',
            f'--gpus {self.gpu_count}',
        ]
        builder.run_command(' '.join(cmd_parts))

        metrics_src = os.path.join(olmes_output_dir, 'metrics-all.jsonl')
        builder.run_command(f'cp {metrics_src} {local_output}')

        builder.rsync_to_gs(
            os.path.dirname(local_output),
            os.path.join(gs_root, 'ModelEvaluationDownstream'),
            delete=False,
            checksum=False,
            contents=True,
        )
        

@dataclass(frozen=True)
class HFModel(Artifact):
    """
    Convert a `PretrainedModel` checkpoint to a HuggingFace-compatible format and upload to GS.
    """
    pretrained_model: PretrainedModel | AnnealedModel | MidtrainedModel | AnnealedModel2 | SFTModel | EWC_SFT

    @property
    def run_name(self) -> str:
        return f"{self.pretrained_model.run_name}-hf"

    @property
    def relpath(self) -> str:
        return f"HFModel/{self.run_name}"

    @property
    def checkpoint_relpath(self) -> str:
        return self.relpath

    @property
    def sft_dataset(self) -> str:
        return self.pretrained_model.sft_dataset
    
    @property
    def pretrain_dataset(self) -> str:
        return self.pretrained_model.pretrain_dataset

    @property
    def seed(self) -> int:
        return self.pretrained_model.seed

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False

        remote_root = cast(str, Project.config.GS_PATH)
        remote_path = os.path.join(remote_root, self.relpath)
        remote_files = G.get_remote_files(subfolder="HFModel")
        return any(f.startswith(remote_path) for f in remote_files)

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "gpus": 1,
            "nodes": 1,
            "cpus-per-task": 4,
            "mem": "256GB",
            "partition": "general",
            "time": "1-00:00:00",
        }

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        gs_root = cast(str, Project.config.GS_PATH)

        src_ckpt_rel = self.pretrained_model.checkpoint_relpath
        local_ckpt_dir = os.path.join(local_root, src_ckpt_rel)
        builder.rsync_from_gs(
            os.path.join(gs_root, src_ckpt_rel),
            local_ckpt_dir,
            delete=False,
            checksum=False,
            skip_existing=False,
            check_exists=True,
            contents=True,
        )

        save_folder = os.path.join(local_root, self.relpath)

        olmo_path = cast(str, Project.config.OLMO_PATH)
        
        tokenizer_dir = os.path.join(olmo_path, "olmo_data", "tokenizers")
        tokenizer_file = "allenai_dolma2.json"
        tokenizer_path = os.path.join(tokenizer_dir, tokenizer_file)

        if isinstance(self.pretrained_model, (MidtrainedModel, SFTModel, EWC_SFT)):
            convert_script = os.path.join(olmo_path, "scripts", "convert_olmo2_to_hf.py")
            cmd_parts = [
                "python",
                convert_script,
                f"--input_dir {local_ckpt_dir}",
                f"--output_dir {save_folder}",
                f"--tokenizer_json_path {tokenizer_path}",
            ]
        else:
            convert_script = os.path.join(olmo_path, "hf_olmo", "convert_olmo_to_hf.py")
            cmd_parts = [
                "python",
                convert_script,
                f"--checkpoint-dir {local_ckpt_dir}",
                f"--destination-dir {save_folder}",
                f"--tokenizer {tokenizer_path}",
                "--keep-olmo-artifacts"
            ]
        
        builder.run_command(" ".join(cmd_parts))

        remote_folder = os.path.join(gs_root, self.relpath)
        builder.upload_to_gs(save_folder, remote_folder, directory=True)