import os
import random
import math
import logging
from dataclasses import dataclass
from typing import cast, Dict, List, Any

from experiments import Artifact, Task, Project  # type: ignore
from launch import globals as G
from launch.utils.olmo_configuration import get_train_config

log = logging.getLogger(__name__)

BILLION = 10**9
MILLION = 10**6

# --- Data Configuration Maps ---

LIST_OF_PRETRAIN_FILES = {
    "c4": {
        "data_paths": [f'c4/train/preprocessed_c4_v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix_gpt-neox-olmo-dolma-v1_5_part-{i:03d}-00000.npy' for i in range(151)],
        "anneal_data_paths": [f'c4/train/preprocessed_c4_v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix_gpt-neox-olmo-dolma-v1_5_part-{i:03d}-00000.npy' for i in range(151, 171)],
        "tokens_per_file": 750 * MILLION,
        "val": {
            "c4-validation": ['c4/val/eval-data_perplexity_v3_small_gptneox20b_c4_en_val_part-0-00000.npy'],
            "tulu-validation": {"data": ['allenai_tulu-3-sft-mixture/val/input_ids_tulu.npy'], "masks": ['allenai_tulu-3-sft-mixture/val/label_mask.npy']},
            "alpaca-validation": {"data": ['tatsu-lab_alpaca/val/input_ids_alpaca.npy'], "masks": ['tatsu-lab_alpaca/val/label_mask.npy']},
            "gsm8k-validation": {"data": ['openai_gsm8k_main/val/input_ids_gsm8k.npy'], "masks": ['openai_gsm8k_main/val/label_mask.npy']},
            "siqa-validation": {"data": ['social_i_qa/val/input_ids.npy'], "masks": ['social_i_qa/val/label_mask.npy']}
        }
    },
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
            "dclm-validation": ['dclm/val/dclm-20m.npy']#, 'dclm/val/dclm-train-20m.npy']
        }
    }
}

LIST_OF_CPT_FILES = {
    "c4": {
        "starcoder": {
            "data_paths": ["starcoder/train/preprocessed_starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00000.npy"],
            "mask_paths": [],
            "val": {"starcoder-validation": ['starcoder/val/preprocessed_starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00001.npy']}
        },
        "tulu": {
            "data_paths": ["allenai_tulu-3-sft-mixture/train/input_ids.npy"],
            "mask_paths": ["allenai_tulu-3-sft-mixture/train/label_mask.npy"],
            "val": {"tulu-validation": {"data": ['allenai_tulu-3-sft-mixture/val/input_ids_tulu.npy'], "masks": ['allenai_tulu-3-sft-mixture/val/label_mask.npy']}}
        },
        "alpaca": {
            "data_paths": ["tatsu-lab_alpaca/train/input_ids.npy"],
            "mask_paths": ["tatsu-lab_alpaca/train/label_mask.npy"],
            "val": {"alpaca-validation": {"data": ['tatsu-lab_alpaca/val/input_ids_alpaca.npy'], "masks": ['tatsu-lab_alpaca/val/label_mask.npy']}}
        },
        "gsm8k": {
            "data_paths": ["openai_gsm8k_main/train/input_ids.npy"],
            "mask_paths": ["openai_gsm8k_main/train/label_mask.npy"],
            "val": {"gsm8k-validation": {"data": ['openai_gsm8k_main/val/input_ids_gsm8k.npy'], "masks": ['openai_gsm8k_main/val/label_mask.npy']}}
        },
        "siqa": {
            "data_paths": ["social_i_qa/train/input_ids.npy"],
            "mask_paths": ["social_i_qa/train/label_mask.npy"],
            "val": {"siqa-validation": {"data": ['social_i_qa/val/input_ids.npy'], "masks": ['social_i_qa/val/label_mask.npy']}}
        },
        "rte": {
            "data_paths": ["SetFit_rte/train/input_ids.npy"],
            "mask_paths": ["SetFit_rte/train/label_mask.npy"],
            "val": {"rte-validation": {"data": ['SetFit_rte/val/input_ids.npy'], "masks": ['SetFit_rte/val/label_mask.npy']}}
        },
        "tinygsm": {
            "data_paths": ["TinyGSM_TinyGSM/train/input_ids.npy"],
            "mask_paths": ["TinyGSM_TinyGSM/train/label_mask.npy"],
            "val": {"tinygsm-validation": {"data": ['TinyGSM_TinyGSM/val/input_ids.npy'], "masks": ['TinyGSM_TinyGSM/val/label_mask.npy']}}
        },
    },
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
        "alpaca": {
            "data_paths": ["tatsu-lab_alpaca/train/input_ids.npy"],
            "mask_paths": ["tatsu-lab_alpaca/train/label_mask.npy"],
            "val": {"alpaca-validation": {"data": ['tatsu-lab_alpaca/val/input_ids-alpaca.npy'], "masks": ['tatsu-lab_alpaca/val/label_mask.npy']}},
            "train_tokens": 3.6,
        },
        "gsm8k": {
            "data_paths": ["openai_gsm8k/train/input_ids.npy"],
            "mask_paths": ["openai_gsm8k/train/label_mask.npy"],
            "val": {"gsm8k-validation": {"data": ['openai_gsm8k/val/input_ids-gsm8k.npy'], "masks": ['openai_gsm8k/val/label_mask.npy']}},
            "train_tokens": 1.2,
        },
        "siqa": {
            "data_paths": ["allenai_social_i_qa/train/input_ids.npy"],
            "mask_paths": ["allenai_social_i_qa/train/label_mask.npy"],
            "val": {"siqa-validation": {"data": ['allenai_social_i_qa/val/input_ids-siqa.npy'], "masks": ['allenai_social_i_qa/val/label_mask.npy']}},
            "train_tokens": 1.18,
        },

    }
}

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
    },
}

WARMUP_STEPS = {
    20: 1000,
    60: 2000,
    150: 3000,
}

ANNEAL_CKPT = {
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
        }
    }
}

# Merge pretrain validation sets into CPT dictionaries
for base in LIST_OF_PRETRAIN_FILES:
    base_val = LIST_OF_PRETRAIN_FILES[base]["val"]
    for cpt in LIST_OF_CPT_FILES.get(base, {}):
        LIST_OF_CPT_FILES[base][cpt]["val"].update(base_val)


# --- Helper Functions ---

def get_train_files(dataset_name: str, n_tokens_billion: int, train_stage: str = 'stable') -> List[str]:
    info = LIST_OF_PRETRAIN_FILES[dataset_name]
    data_source = info["data_paths"] if train_stage == 'stable' else info["anneal_data_paths"]
    
    n_files = min(math.ceil(n_tokens_billion * BILLION / info["tokens_per_file"]) + 1, len(data_source))
    log.info(f"Downloading {n_files} files from {dataset_name.upper()}")
    return data_source[:n_files]


# --- Artifact Classes ---

@dataclass(frozen=True)
class PretrainedModel(Artifact):
    train_tokens: int
    model_size: str = '20m'
    optimizer: str = 'adamw'
    pt_lr: float = 6e-4
    weight_decay: float = 0.1
    sam_rho: float = 0.05
    sam_base_optimizer: str = 'adamw'
    batch_size: int = 256
    scheduler_name: str = 'cosine_with_warmup'
    # scheduler_t_warmup: int = 5000
    scheduler_alpha_f: float = 0.1
    pretrain_gpus: int = 8
    muon_learning_rate: float = 5e-1
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    pretrain_dataset: str = "dclm"

    @property
    def learning_rate(self) -> float:
        if self.pt_lr == -1:
            return PT_LR[int(self.model_size[:-1])][self.train_tokens]
        return self.pt_lr

    @property
    def scheduler_t_warmup(self) -> int:
        return WARMUP_STEPS[int(self.model_size[:-1])]

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
        tk_str = f'{self.train_tokens}B'
        
        name = f'OLMo2-{self.model_size}-tk{tk_str}-{self.optimizer}-lr{lr_str}-wd{wd_str}-bs{self.batch_size}'
        
        if self.optimizer == 'sam':
            name = f'OLMo2-{self.model_size}-tk{tk_str}-sam_{self.sam_base_optimizer}-lr{lr_str}-wd{wd_str}-bs{self.batch_size}'
            name += f'-rho{self.sam_rho:.0e}'.replace('e-0', 'e-')
        elif self.optimizer == 'muon':
            name += f'-muon_lr{self.muon_learning_rate:.0e}'.replace('e-0', 'e-')
            
        return name

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
            
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder='PretrainedModel')
        
        found = any(f.startswith(remote_path) for f in remote_files)
        status = "✓ EXISTS" if found else "❌ NOT found"
        log.info(f"[PretrainedModel] {status}: {self.run_name}")
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "gres": f"gpu:{self.pretrain_gpus}",
            "nodes": 1,
            "cpus": max(1, self.pretrain_gpus * 2),
            "mem": '64GB',
            "requeue": True,
            "partition": 'flame',
            "qos": 'flame-16gpu-c_qos',
            "account": 'aditirag',
            "time": "12-00:00:00"
        }

    def construct(self, builder: Task):
        local_output_path = G.get_random_local_path()
        local_data_path = local_output_path if G.DOWNLOAD_DATA else G.LOCAL_DATA_PATH
        
        save_folder = os.path.join(local_output_path, self.relpath)
        remote_folder = os.path.join(cast(str, G.GS_PATH), self.relpath)
        
        # 1. Prepare Data Paths
        train_data_paths = [os.path.join(local_data_path, p) for p in get_train_files(self.pretrain_dataset, self.train_tokens)]
        
        val_info = LIST_OF_PRETRAIN_FILES[self.pretrain_dataset]["val"]
        eval_datasets = {}
        eval_datasets["pretrain"] = {
            k: [os.path.join(local_data_path, p) for p in v] if isinstance(v, list) else v  # Simplified for brevity
            for k, v in val_info.items()
        }

        # Special handling for dict-based val sets (masks)
        for k, v in val_info.items():
            if isinstance(v, dict):
                eval_datasets["pretrain"][k] = {mk: [os.path.join(local_data_path, p) for p in mv] for mk, mv in v.items()}
        
        eval_datasets["cpt"] = {}

        # 2. Config Overrides
        model_overrides = {}
        if self.pretrain_dataset == "dclm":
            model_overrides = {'vocab_size': 100278, 'embedding_size': 100352, 'eos_token_id': 100257, 'pad_token_id': 100277}

        # 3. Generate Config
        config = get_train_config(
            run_name=self.run_name,
            save_folder=save_folder,
            train_data_paths=train_data_paths,
            eval_datasets=eval_datasets,
            model_size=self.model_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            muon_learning_rate=self.muon_learning_rate,
            muon_momentum=self.muon_momentum,
            muon_weight_decay=self.muon_weight_decay,
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
                'identifier': f'tokenizers/allenai_{"dolma2" if self.pretrain_dataset=="dclm" else "gpt-neox-olmo-dolma-v1_5"}.json',
                'truncate_direction': 'right',
            },
            dtype='uint32' if self.pretrain_dataset == "dclm" else 'uint16',
        )

        # 4. Sync & Execution
        builder.set_env("SYNC_CMD", f"gsutil -m rsync -r {save_folder}/ {remote_folder}/")
        builder.ensure_directory(save_folder)
        
        builder.rsync_from_gs(remote_folder, save_folder, delete=True, checksum=False, skip_existing=False, check_exists=True, contents=True)
        
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(config_path, config)

        # 5. Data Download
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
            f'--nproc_per_node={self.pretrain_gpus} {train_script} {config_path}'
        )

        builder.rsync_to_gs(save_folder, remote_folder, delete=True, checksum=False, contents=True)

@dataclass(frozen=True)
class AnnealedModel(Artifact):
    """
    AnnealedModel loads a checkpoint from a PretrainedModel and continues training
    with the WSD (Warmup-Stable-Decay) scheduler.
    """
    pretrained_model: PretrainedModel
    pretrain_ckpt_step: int  # Step number of the checkpoint to load
    anneal_gpus: int = 8
    anneal_steps: int = None
    anneal_match: str = "token" # "token" or "compute"
    anneal_optim: str = 'adamw'

    @property
    def relpath(self) -> str:
        return f'AnnealedModel/{self.run_name}'

    @property
    def anneal_pretrain_ckpt_step(self) -> int:
        return ANNEAL_CKPT[self.anneal_match][int(self.model_size[:-1])][self.pretrained_model.train_tokens]

    @property
    def checkpoint_relpath(self) -> str:
        return f'{self.relpath}/final-unsharded'

    @property
    def run_name(self) -> str:
        if self.anneal_steps is not None:
            return f'{self.pretrained_model.run_name}-anneal-{self.anneal_optim}-ckpt{self.pretrain_ckpt_step}-steps{self.anneal_steps}'
        if self.anneal_percent is not None:
            if self.model_size == "60m" and self.anneal_optim == "adamw":
                return f'{self.pretrained_model.run_name}-anneal-ckpt{self.pretrain_ckpt_step}-percent{self.anneal_percent}'
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
            "gpus": self.anneal_gpus,
            "nodes": 1,
            "cpus": max(1, self.anneal_gpus * 2),
            "mem": '64GB',
            "requeue": True,
            # "partition": 'preempt',
            "partition": 'general',
            "time": "2-00:00:00"
        }

    def construct(self, builder: Task):

        assert (self.anneal_steps is None or self.anneal_percent is None), "Specify either anneal steps or anneal percent"
        assert not (self.anneal_steps is None and self.anneal_percent is None)
        local_output_path = G.get_random_local_path()
        local_data_path = local_output_path if G.DOWNLOAD_DATA else G.LOCAL_DATA_PATH
        
        save_folder = os.path.join(local_output_path, self.relpath)
        gs_path = cast(str, G.GS_PATH)
        remote_folder = os.path.join(gs_path, self.relpath)

        # 1. Checkpoint Resolution
        # Path format: {pretrain_relpath}/step{step}-unsharded
        ckpt_rel = f'{self.pretrained_model.relpath}/step{self.pretrain_ckpt_step}-unsharded'
        pretrained_model_gs_path = os.path.join(gs_path, ckpt_rel)
        local_checkpoint_path = os.path.join(local_output_path, ckpt_rel)

        log.info(f"Downloading checkpoint from: {pretrained_model_gs_path}")
        builder.rsync_from_gs(pretrained_model_gs_path, local_checkpoint_path, delete=True, checksum=False, skip_existing=False, check_exists=True, contents=True)

        # 2. Annealing Token Calculation (10% of checkpoint tokens)
        # seq_len = 1024
        # tokens_at_ckpt = self.pretrain_ckpt_step * self.pretrained_model.batch_size * seq_len
        # anneal_tokens_billions = (tokens_at_ckpt * 0.1) / BILLION

        # 3. Data Preparation
        if self.anneal_steps is not None:
            anneal_tokens = self.anneal_steps * 1024 * 256 / BILLION
            max_duration = self.anneal_steps
        else:
            anneal_tokens = (self.anneal_percent * self.pretrain_ckpt_step / 100) * 1024 * 256 / BILLION
            max_duration = int(self.anneal_percent * self.pretrain_ckpt_step / 100)

        train_data_paths = [
            os.path.join(local_data_path, p) 
            for p in get_train_files(self.pretrain_dataset, int(anneal_tokens) + 1, train_stage='decay')
        ]

        val_info = LIST_OF_PRETRAIN_FILES[self.pretrain_dataset]["val"]
        eval_datasets = {}
        eval_datasets["pretrain"] = {}
        eval_datasets["cpt"] = {}
        for k, v in val_info.items():
            if isinstance(v, list):
                eval_datasets["pretrain"][k] = [os.path.join(local_data_path, p) for p in v]
            elif isinstance(v, dict):
                eval_datasets["pretrain"][k] = {mk: [os.path.join(local_data_path, p) for p in mv] for mk, mv in v.items()}

        # 4. Config Overrides
        model_overrides = {}
        if self.pretrain_dataset == "dclm":
            model_overrides = {'vocab_size': 100278, 'embedding_size': 100352, 'eos_token_id': 100257, 'pad_token_id': 100277}

        # 5. Generate Configuration
        config = get_train_config(
            run_name=self.run_name,
            save_folder=save_folder,
            train_data_paths=train_data_paths,
            eval_datasets=eval_datasets,
            model_size=self.model_size,
            optimizer=self.anneal_optim,
            learning_rate=self.pretrained_model.learning_rate,
            weight_decay=self.pretrained_model.weight_decay,
            muon_learning_rate=self.pretrained_model.muon_learning_rate,
            muon_momentum=self.pretrained_model.muon_momentum,
            muon_weight_decay=self.pretrained_model.muon_weight_decay,
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
            reset_optimizer_state=False,  # Essential for maintaining momentum in WSD
            restore_dataloader=False,
            try_load_latest_save=True,
            run_sync_cmd=True,
            tokenizer={
                'identifier': f'tokenizers/allenai_{"dolma2" if self.pretrain_dataset=="dclm" else "gpt-neox-olmo-dolma-v1_5"}.json',
                'truncate_direction': 'right',
            },
            dtype='uint32' if self.pretrain_dataset == "dclm" else 'uint16',
        )

        # 6. Execution & Sync
        builder.set_env("SYNC_CMD", f"gsutil -m rsync -r {save_folder}/ {remote_folder}/")
        builder.ensure_directory(save_folder)
        
        # Pull existing results for this specific anneal run if it crashed
        builder.rsync_from_gs(remote_folder, save_folder, delete=True, checksum=False, skip_existing=False, check_exists=True, contents=True)
        
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(config_path, config)

        # 7. Data Downloads
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
class CPTModel(Artifact):
    train_tokens: int
    pretrained_model: PretrainedModel | AnnealedModel
    cpt_dataset: str
    optimizer: str = 'adamw'
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    batch_size: int = 64
    scheduler_name: str = 'cosine_with_warmup'
    scheduler_alpha_f: float = 0.1
    cpt_gpus: int = 1
    muon_learning_rate: float = 5e-4
    muon_weight_decay: float = 0.1
    muon_momentum: float = 0.95

    @property
    def relpath(self) -> str:
        return f'CPTModel/{self.cpt_dataset}/{self.run_name}'

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
        
        name = f'{pre_name}-CPT-{self.cpt_dataset}-tk{self.train_tokens}M-lr{lr_str}-wd{wd_str}-bs{self.batch_size}'
        if self.optimizer == 'muon':
            muon_lr = f'{self.muon_learning_rate:.2e}'.replace('e-0', 'e-')
            name += f'-muon_lr{muon_lr}'
        return name

    @property
    def exists(self) -> bool:
        # return False
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder=f'CPTModel/{self.cpt_dataset}')
        found = any(f.startswith(remote_path) for f in remote_files)
        log.info(f"[CPTModel] {'✓ EXISTS' if found else '❌ NOT found'}: {self.cpt_dataset}")
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {
            'gpus': str(self.cpt_gpus),
            'nodes': 1,
            'cpus-per-task': self.cpt_gpus * 2,
            'mem': '64GB',
            'requeue': True,
            'partition': 'preempt'
        }

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        save_folder = os.path.join(local_root, self.relpath)
        gs_root = cast(str, Project.config.GS_PATH)
        remote_folder = os.path.join(gs_root, self.relpath)

        # 1. Load Pretrained Checkpoint
        pre_ckpt_rel = self.pretrained_model.checkpoint_relpath
        local_pre_path = os.path.join(local_root, pre_ckpt_rel)
        builder.rsync_from_gs(os.path.join(gs_root, pre_ckpt_rel), local_pre_path, delete=True, checksum=True, skip_existing=True, check_exists=True, contents=True, )

        # 2. Map CPT Data
        dataset_info = LIST_OF_CPT_FILES[self.pretrain_dataset][self.cpt_dataset]
        train_paths = [os.path.join(local_root, p) for p in dataset_info["data_paths"]]
        mask_paths = [os.path.join(local_root, p) for p in dataset_info["mask_paths"]]
        tmp_train_tokens = dataset_info.get("train_tokens", None)

        eval_datasets = {}
        eval_datasets["pretrain"] = {}
        eval_datasets["cpt"] = {}
        for k, v in dataset_info["val"].items():
            paths = v.get("data", []) if isinstance(v, dict) else v
            masks = v.get("masks", None) if isinstance(v, dict) else None
            if masks is None:
                eval_datasets["pretrain"][k] = [os.path.join(local_root, p) for p in paths]
            else:
                eval_datasets["cpt"][k] = {
                    "data_paths" : [os.path.join(local_root, p) for p in paths],
                    "mask_paths" : [os.path.join(local_root, m) for m in masks]
                }


        # 3. Step & Scheduler Math
        train_tokens = self.train_tokens
        if tmp_train_tokens is not None:
            log.info(f"Using train tokens from dataset info: {train_tokens}M")
            train_tokens = tmp_train_tokens

        total_tokens = train_tokens * MILLION
        total_steps = max(1, total_tokens // (self.batch_size * 1024))
        warmup_steps = max(1, int(total_steps * 0.1))

        # 4. Config & Overrides
        overrides = {}
        if self.pretrain_dataset == "dclm":
            overrides = {'vocab_size': 100278, 'embedding_size': 100352, 'eos_token_id': 100257, 'pad_token_id': 100277}

        config = get_train_config(
            run_name=self.run_name,
            save_folder=save_folder,
            train_data_paths=train_paths,
            train_data_label_mask_paths=mask_paths,
            # eval_datasets=eval_datasets,
            model_size=self.pretrained_model.model_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            muon_learning_rate=self.muon_learning_rate,
            muon_momentum=self.muon_momentum,
            muon_weight_decay=self.muon_weight_decay,
            max_duration=f'{train_tokens}e6T',
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
            wandb_id=self.run_name,
            load_path=local_pre_path,
            reset_optimizer_state=True,
            tokenizer={'identifier': f'tokenizers/allenai_{"dolma2" if self.pretrain_dataset=="dclm" else "gpt-neox-olmo-dolma-v1_5"}.json'},
            dtype='uint32' if self.pretrain_dataset == "dclm" else 'uint16',
        )

        # 5. Execute
        builder.ensure_directory(save_folder)
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(os.path.join(save_folder, 'config.yaml'), config)

        # Download all data (Train + Masks + Eval)
        gs_data_path = cast(str, Project.config.GS_DATA_PATH)
        for p in (train_paths + mask_paths + [item for sub in eval_datasets["pretrain"].values() for item in sub] + [item for sub_dict in eval_datasets["cpt"].values() for sub_list in sub_dict.values() for item in sub_list]):
            builder.download_from_gs(p.replace(local_root, gs_data_path), p, directory=False)

        olmo_path = cast(str, Project.config.OLMO_PATH)
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && '
            f'torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d --nproc_per_node={self.cpt_gpus} {train_script} {config_path}'
        )
        builder.upload_to_gs(save_folder, remote_folder, directory=True)


@dataclass(frozen=True)
class PerturbedModel(Artifact):
    base_model: "PretrainedModel | AnnealedModel"
    sigma: float
    seed: int = 64
    device: str = "cpu"

    @staticmethod
    def _format_sigma(sigma: float) -> str:
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
        # return False
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.checkpoint_relpath)
        remote_files = G.get_remote_files(subfolder="PerturbedModel")
        found = any(f.startswith(remote_path) for f in remote_files)
        log.info(f"[PerturbedModel] {'✓ EXISTS' if found else '❌ NOT found'}: {self.run_name}")
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {
            "gpus": 1,
            "nodes": 1,
            "cpus-per-task": 4,
            "mem": "64GB",
            "requeue": True,
            "partition": "preempt",
            "time": "2-00:00:00",
        }

    def construct(self, builder: Task):
        gs_root = cast(str, Project.config.GS_PATH)
        if isinstance(self.base_model, PretrainedModel):
            base_subfolder = "PretrainedModel"
        elif isinstance(self.base_model, AnnealedModel):
            base_subfolder = "AnnealedModel"
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
    model: "PretrainedModel | CPTModel | AnnealedModel | PerturbedModel"
    device: str = 'cuda'
    chunk_size: int = 1024
    hf_model: bool = False
    quant_bit: int = None

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
        # return False  # Force re-eval

    def get_requirements(self) -> Dict[str, Any]:
        return {'gpus': 1, 'nodes': 1, 'cpus-per-task': 4, 'mem': '64GB', 'partition': 'preempt', 'time': "1-00:00:00"}

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        local_output = os.path.join(local_root, self.relpath)
        builder.ensure_directory(os.path.dirname(local_output))

        # 1. Fetch Model
        local_model = os.path.join(local_root, self.model.checkpoint_relpath)
        builder.rsync_from_gs(os.path.join(cast(str, Project.config.GS_PATH), self.model.checkpoint_relpath), local_model, delete=False, checksum=False, skip_existing=False, check_exists=True, contents=True)

        # 2. Gather Eval Data (Unified Logic)
        eval_meta = (LIST_OF_CPT_FILES[self.model.pretrain_dataset][self.model.cpt_dataset]["val"] 
                     if isinstance(self.model, CPTModel) else 
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

        # 3. Run Command
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
        if self.model.pretrain_dataset == "dclm":
            cmd.append(f"--dtype=uint32")
        if self.hf_model:
            cmd.append("--hf_model")
            if self.quant_bit is not None:
                cmd.append(f"--quantize={self.quant_bit}")
        
        builder.run_command(' '.join(cmd))
        builder.rsync_to_gs(os.path.dirname(local_output), os.path.join(cast(str, Project.config.GS_PATH), 'ModelEvaluation'), delete=False, checksum=False, contents=True)

@dataclass(frozen=True)
class HFModel(Artifact):
    """
    Convert a `PretrainedModel` checkpoint to a HuggingFace-compatible format and upload to GS.
    """
    pretrained_model: PretrainedModel | AnnealedModel

    @property
    def run_name(self) -> str:
        # Append `-hf` to the base pretrained run name.
        return f"{self.pretrained_model.run_name}-hf"

    @property
    def relpath(self) -> str:
        return f"HFModel/{self.run_name}"

    @property
    def checkpoint_relpath(self) -> str:
        # Directory in GS where the converted HF checkpoint will live.
        return self.relpath
    
    @property
    def pretrain_dataset(self) -> str:
        return self.pretrained_model.pretrain_dataset

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False

        remote_root = cast(str, Project.config.GS_PATH)
        remote_path = os.path.join(remote_root, self.relpath)
        remote_files = G.get_remote_files(subfolder="HFModel")
        return any(f.startswith(remote_path) for f in remote_files)

    def get_requirements(self) -> Dict[str, Any]:
        # Similar footprint to evaluation / light conversion.
        return {
            "gpus": 1,
            "nodes": 1,
            "cpus-per-task": 4,
            "mem": "64GB",
            "partition": "preempt",
            "time": "1-00:00:00",
        }

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        gs_root = cast(str, Project.config.GS_PATH)

        # 1. Download the pretrained final-unsharded checkpoint locally.
        src_ckpt_rel = self.pretrained_model.checkpoint_relpath  # e.g. PretrainedModel/<run>/final-unsharded
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

        # 2. Prepare local output directory for the converted HF checkpoint.
        save_folder = os.path.join(local_root, self.relpath)
        # builder.ensure_directory(save_folder)

        # 3. Run the HF conversion script with the appropriate tokenizer.
        olmo_path = cast(str, Project.config.OLMO_PATH)
        convert_script = os.path.join(olmo_path, "hf_olmo", "convert_olmo_to_hf.py")

        # Select tokenizer JSON based on the pretrain dataset.
        tokenizer_dir = os.path.join(olmo_path, "olmo_data", "tokenizers")
        if self.pretrain_dataset == "dclm":
            tokenizer_file = "allenai_dolma2.json"
        else:
            tokenizer_file = "allenai_gpt-neox-olmo-dolma-v1_5.json"
        tokenizer_path = os.path.join(tokenizer_dir, tokenizer_file)

        cmd_parts = [
            "python",
            convert_script,
            f"--checkpoint-dir {local_ckpt_dir}",
            f"--destination-dir {save_folder}",
            f"--tokenizer {tokenizer_path}",
            "--keep-olmo-artifacts"
        ]
        builder.run_command(" ".join(cmd_parts))

        # 4. Upload the converted HF model back to GS.
        remote_folder = os.path.join(gs_root, self.relpath)
        builder.upload_to_gs(save_folder, remote_folder, directory=True)


# @dataclass(frozen=True)
# class QuantizedModel(Artifact):
#     """
#     Quantize a `HFModel` checkpoint using GPTQ and upload to GS.
#     """
#     hf_model: HFModel
#     bits: int

#     @property
#     def run_name(self) -> str:
#         # Append `quant-{bits}bit` to the base HF model run name.
#         return f"{self.hf_model.run_name}-quant-{self.bits}bit"

#     @property
#     def relpath(self) -> str:
#         return f"QuantizedModel/{self.run_name}"

#     @property
#     def checkpoint_relpath(self) -> str:
#         # Directory in GS where the quantized checkpoint will live.
#         return self.relpath

#     @property
#     def pretrain_dataset(self) -> str:
#         return self.hf_model.pretrain_dataset

#     @property
#     def exists(self) -> bool:
#         return False
#         if not Project.config.CHECK_EXISTS_REMOTE:
#             return False

#         remote_root = cast(str, Project.config.GS_PATH)
#         remote_path = os.path.join(remote_root, self.relpath)
#         remote_files = G.get_remote_files(subfolder="QuantizedModel")
#         return any(f.startswith(remote_path) for f in remote_files)

#     def get_requirements(self) -> Dict[str, Any]:
#         # Quantization can be memory-intensive, similar to HF conversion.
#         return {
#             "gpus": 1,
#             "nodes": 1,
#             "cpus-per-task": 4,
#             "mem": "64GB",
#             "partition": "general",
#             "time": "1-00:00:00",
#         }

#     def construct(self, builder: Task):
#         local_root = G.get_random_local_path()
#         gs_root = cast(str, Project.config.GS_PATH)
#         local_data_path = local_root if G.DOWNLOAD_DATA else G.LOCAL_DATA_PATH

#         # 1. Download the HF model checkpoint locally.
#         src_ckpt_rel = self.hf_model.checkpoint_relpath  # e.g. HFModel/<run>-hf
#         local_hf_dir = os.path.join(local_root, src_ckpt_rel)
#         builder.rsync_from_gs(
#             os.path.join(gs_root, src_ckpt_rel),
#             local_hf_dir,
#             delete=False,
#             checksum=False,
#             skip_existing=False,
#             check_exists=True,
#             contents=True,
#         )

#         # 2. Download calibration dataset (first anneal_data_path)
#         pretrain_info = LIST_OF_PRETRAIN_FILES[self.pretrain_dataset]
#         calibration_data_path_relative = pretrain_info["anneal_data_paths"][0]
#         calibration_data_path = os.path.join(local_data_path, calibration_data_path_relative)
        
#         if G.DOWNLOAD_DATA:
#             gs_data_path = cast(str, G.GS_DATA_PATH)
#             gs_calibration_path = os.path.join(gs_data_path, calibration_data_path_relative)
#             # Ensure parent directory exists
#             builder.ensure_directory(os.path.dirname(calibration_data_path))
#             builder.download_from_gs(gs_calibration_path, calibration_data_path, directory=False)

#         # 3. Prepare local output directory for the quantized model.
#         save_folder = os.path.join(local_root, self.relpath)
#         builder.ensure_directory(save_folder)

#         # 4. Run the quantization script.
#         olmo_path = cast(str, Project.config.OLMO_PATH)
#         quantize_script = os.path.join(olmo_path, "new_utils", "quantize_model.py")

#         cmd_parts = [
#             "python",
#             quantize_script,
#             f"--bits {self.bits}",
#             f"--model-dir {local_hf_dir}",
#             f"--quantized-dir {save_folder}",
#             f"--calibration-data-path {calibration_data_path}",
#             f"--pretrain-dataset {self.pretrain_dataset}",
#         ]
#         builder.run_command(" ".join(cmd_parts))

#         # 5. Upload the quantized model back to GS.
#         remote_folder = os.path.join(gs_root, self.relpath)
#         builder.upload_to_gs(save_folder, remote_folder, directory=True)
