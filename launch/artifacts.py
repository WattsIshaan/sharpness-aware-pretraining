import os
import random
import math
import json
import logging
import shutil
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
    },
    # "dolmino": {
    #     "val": {
    #         "dclm-validation": ['dclm/val/dclm-20m.npy']#, 'dclm/val/dclm-train-20m.npy']
    #     }
    # }
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
        "open-platypus": {
            "data_paths": ["garage-bAInd_Open-Platypus/train/input_ids.npy"],
            "mask_paths": ["garage-bAInd_Open-Platypus/train/label_mask.npy"],
            "val": {"open-platypus-validation": {"data": ['garage-bAInd_Open-Platypus/val/input_ids-open-platypus.npy'], "masks": ['garage-bAInd_Open-Platypus/val/label_mask.npy']}},
            "train_tokens": 7,
        },
        "stackmathqa": {
            "data_paths": ["math-ai_StackMathQA/train/input_ids.npy"],
            "mask_paths": ["math-ai_StackMathQA/train/label_mask.npy"],
            "val": {"stackmathqa-validation": {"data": ['math-ai_StackMathQA/val/input_ids-stackmathqa.npy'], "masks": ['math-ai_StackMathQA/val/label_mask.npy']}},
        },
        "helpsteer": {
            "data_paths": ["nvidia_HelpSteer/train/input_ids.npy"],
            "mask_paths": ["nvidia_HelpSteer/train/label_mask.npy"],
            "val": {"helpsteer-validation": {"data": ['nvidia_HelpSteer/val/input_ids-helpsteer.npy'], "masks": ['nvidia_HelpSteer/val/label_mask.npy']}},
        },

    },
    "dolmino": {
        "tulu": {
            "data_paths": ["allenai_tulu-3-sft-mixture/train/input_ids.npy"],
            "mask_paths": ["allenai_tulu-3-sft-mixture/train/label_mask.npy"],
            "val": {"tulu-validation": {"data": ['allenai_tulu-3-sft-mixture/val/input_ids-tulu.npy'], "masks": ['allenai_tulu-3-sft-mixture/val/label_mask.npy']}},
        },
        "starcoder": {
            "data_paths": ["bigcode_starcoderdata/train/input_ids.npy"],
            "mask_paths": ["bigcode_starcoderdata/train/label_mask.npy"],
            "val": {"starcoder-validation": {"data": ['bigcode_starcoderdata/val/input_ids-starcoder.npy'], "masks": ['bigcode_starcoderdata/val/label_mask.npy']}},
        },
        "musicpile": {
            "data_paths": ["musicpile/train/input_ids.npy"],
            "mask_paths": ["musicpile/train/label_mask.npy"],
            "val": {"musicpile-validation": {"data": ['musicpile/val/input_ids-musicpile.npy'], "masks": ['musicpile/val/label_mask.npy']}},
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
        "open-platypus": {
            "data_paths": ["garage-bAInd_Open-Platypus/train/input_ids.npy"],
            "mask_paths": ["garage-bAInd_Open-Platypus/train/label_mask.npy"],
            "val": {"open-platypus-validation": {"data": ['garage-bAInd_Open-Platypus/val/input_ids-open-platypus.npy'], "masks": ['garage-bAInd_Open-Platypus/val/label_mask.npy']}},
            "train_tokens": 7,
        },
        "stackmathqa": {
            "data_paths": ["math-ai_StackMathQA/train/input_ids.npy"],
            "mask_paths": ["math-ai_StackMathQA/train/label_mask.npy"],
            "val": {"stackmathqa-validation": {"data": ['math-ai_StackMathQA/val/input_ids-stackmathqa.npy'], "masks": ['math-ai_StackMathQA/val/label_mask.npy']}},
        },
        "helpsteer": {
            "data_paths": ["nvidia_HelpSteer/train/input_ids.npy"],
            "mask_paths": ["nvidia_HelpSteer/train/label_mask.npy"],
            "val": {"helpsteer-validation": {"data": ['nvidia_HelpSteer/val/input_ids-helpsteer.npy'], "masks": ['nvidia_HelpSteer/val/label_mask.npy']}},
        },
        "codealpaca": {
            "data_paths": ["HuggingFaceH4_CodeAlpaca_20K/train/input_ids.npy"],
            "mask_paths": ["HuggingFaceH4_CodeAlpaca_20K/train/label_mask.npy"],
            "val": {"codealpaca-validation": {"data": ['HuggingFaceH4_CodeAlpaca_20K/val/input_ids-codealpaca.npy'], "masks": ['HuggingFaceH4_CodeAlpaca_20K/val/label_mask.npy']}},
        },
    },
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

ANNEAL_CKPT2 = {
    60: {
        5: 695000,
        10: 660000,
        20: 585000,
        50: 365000,
        100: 0,
    },
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
    sequence_length: int = 1024

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
        # return False
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
            "gpus": self.pretrain_gpus,
            "nodes": 1,
            "cpus": self.pretrain_gpus * 2,
            "mem": '64GB',
            "requeue": True,
            "time": "2-00:00:00",
            "partition": 'general',
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

        # Cleanup local directory
        # log.info(f"Cleaning up local directory: {local_output_path}")
        # shutil.rmtree(local_output_path, ignore_errors=True)

@dataclass(frozen=True)
class MidtrainedModel(Artifact):
    """
    MidtrainedModel for OLMo2-1B stage-2 (midtraining from a pre-trained checkpoint).
    Uses hyperparameters from configs/official-0425/OLMo2-1B-stage2-seed42.yaml.
    Data paths are loaded from launch/utils/midtrain_data.json.
    """
    optimizer: str = 'adamw'
    sam_rho: float = 0.05
    train_tokens: int = 4
    midtrain_gpus: int = 8
    global_train_batch_size: int = 1024
    midtrain_tokens: int = 5
    seed: int = 42
    per_device_train_batch_size: int = 8
    anneal_sam: bool = False
    sam_per_microbatch: bool = False
    sequence_length: int = 2048
    scheduler_alpha_f: float = 0.0

    @property
    def relpath(self) -> str:
        return f'MidtrainedModel/{self.run_name}'

    @property
    def run_name(self) -> str:
        name = f'OLMo2-1b-tk{self.train_tokens}T-adamw-Midtrain-{self.midtrain_tokens}B-{self.optimizer}'
        if self.sam_per_microbatch:
            name += '_per_microbatch'
        if self.optimizer == "sam":
            if self.sam_rho == 0.15:
                name += f'-rho{self.sam_rho:.2e}'.replace('e-0', 'e-')
            else:
                name += f'-rho{self.sam_rho:.0e}'.replace('e-0', 'e-')
            if self.anneal_sam:
                name += '-anneal'
        name += f'-bs{self.global_train_batch_size}'
        if self.scheduler_alpha_f != 0.0:
            name += f'-alpha_f{self.scheduler_alpha_f:.2e}'.replace('e-0', 'e-')
        return name

    @property
    def checkpoint_relpath(self) -> str:
        return f'{self.relpath}/final-unsharded'
    
    @property
    def load_path(self) -> str:
        if self.train_tokens == 4:
            return 'PretrainedModel/OLMo2-1b-tk4T-adamw/final-unsharded'
        elif self.train_tokens == 2:
            return 'PretrainedModel/OLMo2-1b-tk4T-adamw/step950000-unsharded'
        else:
            raise ValueError(f"Invalid train_tokens: {self.train_tokens}")
    
    @property
    def learning_rate(self) -> float:
        if self.train_tokens == 2:
            return 2.7699e-4
        elif self.train_tokens == 4:
            return 7.4487e-5
        else:
            raise ValueError(f"Invalid train_tokens: {self.train_tokens}")
    
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
        local_output_path = G.get_random_local_path()
        local_data_path = local_output_path if G.DOWNLOAD_DATA else cast(str, G.LOCAL_DATA_PATH)
        save_folder = os.path.join(local_output_path, self.relpath)
        gs_path = cast(str, G.GS_PATH)
        remote_folder = os.path.join(gs_path, self.relpath)
        olmo_path = cast(str, Project.config.OLMO_PATH)

        pre_ckpt_rel = self.load_path
        local_pre_path = os.path.join(local_output_path, pre_ckpt_rel)
        builder.rsync_from_gs(os.path.join(gs_path, pre_ckpt_rel), local_pre_path, delete=True, checksum=True, skip_existing=True, check_exists=True, contents=True)

        # 3. Resolve train/eval paths: if gs:// and DOWNLOAD_DATA, download to local
        dataset_info = json.load(open(os.path.join(olmo_path, 'launch', 'utils', 'midtrain_data.json')))
        train_data_paths = [os.path.join(local_data_path, p) for p in dataset_info['train_data_paths']]
        # train_data_paths = [train_data_paths[0]]

        # Build eval_datasets for get_train_config: pretrain-perplexity style
        eval_datasets = None
        eval_datasets = {'downstream': [
            'winogrande',
            'mmlu_other_var',
            'sciq',
            'hellaswag',
            'copa',
            'openbook_qa',
        ]}

        # 4. Generate config (1B midtraining hyperparameters from OLMo2-1B-stage2-seed42.yaml)
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
            # stop_at=23852,
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
            anneal_sam=self.anneal_sam,
            sam_per_microbatch=self.sam_per_microbatch,
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
            # sharded_checkpointer='olmo_core',
            distributed_strategy='ddp',
            gen1_gc_interval=10,
            # fsdp={
            #     'wrapping_strategy': None,
            #     'sharding_strategy': 'SHARD_GRAD_OP',
            #     'precision': 'mixed',
            # },
        )
        # Overrides for midtraining-specific options
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
        # 5. Sync & Execution
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
            # for local_dir in all_paths:
                gs_dir = local_dir.replace(local_data_path, gs_data_path)
                builder.download_from_gs(gs_dir, local_dir, directory=True)
                # builder.download_from_gs(gs_dir, local_dir, directory=False, contents=True)

        
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d '
            f'--nproc_per_node={self.midtrain_gpus} {train_script} {config_path}'
        )

        builder.rsync_to_gs(save_folder, remote_folder, delete=True, checksum=False, contents=True)

        # Cleanup local directory
        # log.info(f"Cleaning up local directory: {local_output_path}")
        # shutil.rmtree(local_output_path, ignore_errors=True)


@dataclass(frozen=True)
class AnnealedModel(Artifact):
    """
    AnnealedModel loads a checkpoint from a PretrainedModel and continues training
    with the WSD (Warmup-Stable-Decay) scheduler.
    """
    pretrained_model: PretrainedModel
    pt_token: int
    anneal_gpus: int = 8
    anneal_steps: int = None
    anneal_match: str = "token" # "token" or "compute"
    anneal_optim: str = 'adamw'
    anneal_percent: int = None

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
    def train_tokens(self) -> int:
        return self.pretrained_model.train_tokens

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
            anneal_tokens = self.anneal_steps * self.sequence_length * 256 / BILLION
            max_duration = self.anneal_steps
        else:
            anneal_tokens = (self.anneal_percent * self.pretrain_ckpt_step / 100) * self.sequence_length * 256 / BILLION
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

        # Cleanup local directory
        # log.info(f"Cleaning up local directory: {local_output_path}")
        # shutil.rmtree(local_output_path, ignore_errors=True)


@dataclass(frozen=True)
class AnnealedModel2(Artifact):
    """
    AnnealedModel loads a checkpoint from a PretrainedModel and continues training
    with the WSD (Warmup-Stable-Decay) scheduler.
    """
    pretrained_model: PretrainedModel
    pt_token: int
    anneal_gpus: int = 8
    anneal_optim: str = 'adamw'
    anneal_percent: int = None

    @property
    def sequence_length(self) -> int:
        return self.pretrained_model.sequence_length

    @property
    def relpath(self) -> str:
        return f'AnnealedModel/{self.run_name}'

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
        remote_files = G.get_remote_files(subfolder='AnnealedModel')
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

        train_data_paths = [os.path.join(local_data_path, p) for p in get_train_files(self.pretrain_dataset, self.train_tokens)]

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
            reset_optimizer_state=False,  # Essential for maintaining momentum in WSD
            restore_dataloader=True,
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

        # Cleanup local directory
        # log.info(f"Cleaning up local directory: {local_output_path}")
        # shutil.rmtree(local_output_path, ignore_errors=True)


@dataclass(frozen=True)
class CPTModel(Artifact):
    train_tokens: int
    pretrained_model: PretrainedModel | AnnealedModel | AnnealedModel2 | MidtrainedModel
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
    use_checkpoint_cache: bool = True
    step: str = 'final'

    @property
    def sequence_length(self) -> int:
        return self.pretrained_model.sequence_length

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
        
        if self.pretrained_model.model_size == '60m' and self.cpt_dataset == 'starcoder' and self.train_tokens == 10:
            name = f'{pre_name}-CPT-{self.cpt_dataset}-lr{lr_str}-wd{wd_str}-bs{self.batch_size}'
        else:
            name = f'{pre_name}-CPT-{self.cpt_dataset}-tk{self.train_tokens}M-lr{lr_str}-wd{wd_str}-bs{self.batch_size}'
        if self.optimizer == 'muon':
            muon_lr = f'{self.muon_learning_rate:.2e}'.replace('e-0', 'e-')
            name += f'-muon_lr{muon_lr}'
        if self.step != 'final':
            name += f'-{self.step}'
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
            'gpus': f"A100_40GB:{str(self.cpt_gpus)}",
            # 'gpus': f"L40:{str(self.cpt_gpus)}",
            'nodes': 1,
            # 'cpus-per-task': self.cpt_gpus * 2,
            'cpus': self.cpt_gpus * 2,
            'mem': '64GB',
            'requeue': True,
            # 'partition': 'preempt',
            "partition": 'general',
            "time": "2-00:00:00"
        }

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        save_folder = os.path.join(local_root, self.relpath)
        gs_root = cast(str, Project.config.GS_PATH)
        remote_folder = os.path.join(gs_root, self.relpath)

        # 1. Load Pretrained Checkpoint
        if self.step != 'final':
            pre_ckpt_rel = f"{self.pretrained_model.relpath}/{self.step}-unsharded"
        else:
            pre_ckpt_rel = self.pretrained_model.checkpoint_relpath
        if self.use_checkpoint_cache:
            pre_root = G.get_cpt_checkpoint_cache_dir()
        else:
            pre_root = local_root
        local_pre_path = os.path.join(pre_root, pre_ckpt_rel)
        builder.rsync_from_gs(os.path.join(gs_root, pre_ckpt_rel), local_pre_path, delete=True, checksum=True, skip_existing=True, check_exists=True, contents=True, )

        # 2. Map CPT Data
        dataset_info = LIST_OF_CPT_FILES[self.pretrain_dataset][self.cpt_dataset]
        train_paths = [os.path.join(local_root, p) for p in dataset_info["data_paths"]]
        mask_paths = [os.path.join(local_root, p) for p in dataset_info["mask_paths"]]
        tmp_train_tokens = dataset_info.get("train_tokens", None)

        eval_datasets = {}
        eval_datasets["pretrain"] = {}
        eval_datasets["cpt"] = {}
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
                eval_datasets["cpt"][k] = {
                    "data_paths" : [os.path.join(local_root, p) for p in paths],
                    "mask_paths" : [os.path.join(local_root, m) for m in masks]
                }
        
        if self.pretrain_dataset == "dolmino":
            del eval_datasets["pretrain"]


        # 3. Step & Scheduler Math
        train_tokens = self.train_tokens
        if tmp_train_tokens is not None:
            log.info(f"Using train tokens from dataset info: {train_tokens}M")
            train_tokens = tmp_train_tokens

        total_tokens = train_tokens * MILLION
        total_steps = max(1, total_tokens // (self.batch_size * self.sequence_length))
        warmup_steps = max(1, int(total_steps * 0.1))

        # 4. Config & Overrides
        overrides = {}
        if self.pretrain_dataset == "dclm":
            overrides = {'vocab_size': 100278, 'embedding_size': 100352, 'eos_token_id': 100257, 'pad_token_id': 100277}

        exp_name = self.run_name[len(self.run_name)-64+1:] if len(self.run_name) > 64 else self.run_name

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
            tokenizer={'identifier': f'tokenizers/allenai_{"dolma2" if self.pretrain_dataset in ("dclm", "dolmino") else "gpt-neox-olmo-dolma-v1_5"}.json'},
            dtype='uint32' if self.pretrain_dataset in ("dclm", "dolmino") else 'uint16',
        )

        # 5. Execute
        builder.ensure_directory(save_folder)
        config_path = os.path.join(save_folder, 'config.yaml')
        builder.create_yaml_file(os.path.join(save_folder, 'config.yaml'), config)

        # Download all data (Train + Masks + Eval)
        gs_data_path = cast(str, Project.config.GS_DATA_PATH)            
        for p in (train_paths + mask_paths + [item for sub_dict in eval_datasets["cpt"].values() for sub_list in sub_dict.values() for item in sub_list]):
            builder.download_from_gs(p.replace(local_root, gs_data_path), p, directory=False)
        if self.pretrain_dataset != "dolmino":
            for p in [item for sub in eval_datasets["pretrain"].values() for item in sub]:
                builder.download_from_gs(p.replace(local_root, gs_data_path), p, directory=False)

        olmo_path = cast(str, Project.config.OLMO_PATH)
        train_script = os.path.join(olmo_path, 'scripts', 'train.py')
        builder.run_command(
            f'cd {olmo_path} && '
            f'torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d --nproc_per_node={self.cpt_gpus} {train_script} {config_path}'
        )
        builder.upload_to_gs(save_folder, remote_folder, directory=True)

        # Cleanup local directory
        # log.info(f"Cleaning up local directory: {local_root}")
        # shutil.rmtree(local_root, ignore_errors=True)


@dataclass(frozen=True)
class PerturbedModel(Artifact):
    base_model: "PretrainedModel | AnnealedModel | MidtrainedModel | AnnealedModel2"
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
    model: "PretrainedModel | CPTModel | AnnealedModel | PerturbedModel | MidtrainedModel | AnnealedModel2"
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
        # return False
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.relpath)
        remote_files = G.get_remote_files(subfolder='ModelEvaluation')
        found = any(f.startswith(remote_path) for f in remote_files)
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {'gres': 'gpu:1', 'nodes': 1, 'cpus': 4, 'mem': '256GB', 'partition': 'general', 'time': '1-00:00:00'}

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
        if self.model.pretrain_dataset == "dclm" or self.model.pretrain_dataset == "dolmino":
            cmd.append(f"--dtype=uint32")
        if self.hf_model:
            cmd.append("--hf_model")
            if self.quant_bit is not None:
                cmd.append(f"--quantize={self.quant_bit}")
        
        builder.run_command(' '.join(cmd))
        builder.rsync_to_gs(os.path.dirname(local_output), os.path.join(cast(str, Project.config.GS_PATH), 'ModelEvaluation'), delete=False, checksum=False, contents=True)

        # Cleanup local directory
        # log.info(f"Cleaning up local directory: {local_root}")
        # shutil.rmtree(local_root, ignore_errors=True)


@dataclass(frozen=True)
class ModelEvaluationDownstream(Artifact):
    """
    Evaluate an HFModel on downstream tasks using the olmes evaluation framework.
    """
    model: "HFModel | SFTModel"
    tasks: tuple = ('core_9mcqa::olmes', 'mmlu:mc::olmes', 'olmo_2_generative::olmes', 'olmo_2_heldout::olmes')
    batch_size: int = 8
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    gpu_count: int = 1

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
        # return False
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

        # 1. Download HFModel checkpoint locally
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

        # 2. Build olmes evaluation command
        olmes_output_dir = os.path.join(local_root, f'{self.run_name}_output')
        builder.ensure_directory(olmes_output_dir)

        # Build model-args string for olmes
        model_args_parts = [
            f'model_path={local_model_path}',
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

        # 3. Copy metrics-all.jsonl to the expected output location
        metrics_src = os.path.join(olmes_output_dir, 'metrics-all.jsonl')
        builder.run_command(f'cp {metrics_src} {local_output}')

        # 4. Upload results to GS
        builder.rsync_to_gs(
            os.path.dirname(local_output),
            os.path.join(gs_root, 'ModelEvaluationDownstream'),
            delete=False,
            checksum=False,
            contents=True,
        )

        # Cleanup local directory
        # log.info(f"Cleaning up local directory: {local_root}")
        # shutil.rmtree(local_root, ignore_errors=True)


@dataclass(frozen=True)
class ModelEvaluationDownstreamOLMo(Artifact):
    """
    Evaluate a model on downstream tasks using the OLMo-native evaluation
    script (new_utils/evaluate_downstream.py) instead of olmes.
    Works with both OLMo 1 and OLMo 2 checkpoints.
    """
    model: "PretrainedModel | CPTModel | AnnealedModel | MidtrainedModel | AnnealedModel2"
    tasks: tuple = ('winogrande', 'mmlu_other_var', 'sciq', 'hellaswag', 'copa', 'openbook_qa')
    batch_size: int = 8
    subset_num_batches: int = 0
    hf_model: bool = False
    quant_bit: int = None

    @property
    def run_name(self) -> str:
        name = self.model.run_name
        if self.quant_bit is not None:
            name += f'-quant-{self.quant_bit}bit'
        return name

    @property
    def relpath(self) -> str:
        return f'ModelEvaluationDownstreamOLMo/{self.run_name}-downstream-eval.json'

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.relpath)
        remote_files = G.get_remote_files(subfolder='ModelEvaluationDownstreamOLMo')
        found = any(f.startswith(remote_path) for f in remote_files)
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {'gres': 'gpu:1', 'nodes': 1, 'cpus': 4, 'mem': '256GB', 'partition': 'general', 'time': '1-00:00:00'}

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        gs_root = cast(str, Project.config.GS_PATH)
        local_output = os.path.join(local_root, self.relpath)
        builder.ensure_directory(os.path.dirname(local_output))

        # 1. Fetch Model
        local_model = os.path.join(local_root, self.model.checkpoint_relpath)
        builder.rsync_from_gs(
            os.path.join(gs_root, self.model.checkpoint_relpath),
            local_model,
            delete=False, checksum=False, skip_existing=False, check_exists=True, contents=True,
        )

        # 2. Run downstream evaluation
        olmo_path = cast(str, Project.config.OLMO_PATH)
        eval_script = os.path.join(olmo_path, 'new_utils', 'evaluate_downstream.py')
        tasks_str = ','.join(self.tasks)
        cmd = [
            'python', eval_script,
            f'--model_path {local_model}',
            f'--tasks {tasks_str}',
            f'--output_path {local_output}',
            f'--device cuda',
            f'--batch_size {self.batch_size}',
            f'--subset_num_batches {self.subset_num_batches}',
        ]
        if self.hf_model:
            cmd.append('--hf_model')
            if self.quant_bit is not None:
                cmd.append(f'--quantize={self.quant_bit}')

        builder.run_command(' '.join(cmd))
        builder.rsync_to_gs(os.path.dirname(local_output), os.path.join(cast(str, Project.config.GS_PATH), 'ModelEvaluationDownstreamOLMo'), delete=False, checksum=False, contents=True)

        log.info(f"Cleaning up local directory: {local_root}")
        shutil.rmtree(local_root, ignore_errors=True)


@dataclass(frozen=True)
class HFModel(Artifact):
    """
    Convert a `PretrainedModel` checkpoint to a HuggingFace-compatible format and upload to GS.
    """
    pretrained_model: PretrainedModel | AnnealedModel | MidtrainedModel | AnnealedModel2 | CPTModel

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
    def seed(self) -> int:
        return self.pretrained_model.seed

    @property
    def exists(self) -> bool:
        # return False
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
            "mem": "256GB",
            "partition": "general",
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
        
        # Select tokenizer JSON based on the pretrain dataset.
        tokenizer_dir = os.path.join(olmo_path, "olmo_data", "tokenizers")
        if self.pretrain_dataset == "dclm" or self.pretrain_dataset == "dolmino":
            tokenizer_file = "allenai_dolma2.json"
        else:
            tokenizer_file = "allenai_gpt-neox-olmo-dolma-v1_5.json"
        tokenizer_path = os.path.join(tokenizer_dir, tokenizer_file)

        if isinstance(self.pretrained_model, MidtrainedModel) or isinstance(self.pretrained_model, CPTModel):
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

        # 4. Upload the converted HF model back to GS.
        remote_folder = os.path.join(gs_root, self.relpath)
        builder.upload_to_gs(save_folder, remote_folder, directory=True)

        # Cleanup local directory
        # log.info(f"Cleaning up local directory: {local_root}")
        # shutil.rmtree(local_root, ignore_errors=True)


@dataclass(frozen=True)
class SFTModel(Artifact):
    """
    Supervised Fine-Tuning of an HFModel using open-instruct's finetune.py.
    """
    hf_model: HFModel
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    lr_scheduler_type: str = 'linear'
    warmup_ratio: float = 0.03
    sft_gpus: int = 8
    seed: int = 1
    dataset_mixer: tuple = ('allenai/tulu-3-sft-olmo-2-mixture-0225', '1.0')
    add_bos: bool = True
    chat_template_name: str = 'tulu'

    @property
    def run_name(self) -> str:
        lr_str = f'{self.learning_rate:.0e}'.replace('e-0', 'e-')
        return f'{self.hf_model.pretrained_model.run_name}-SFT-lr{lr_str}'

    @property
    def relpath(self) -> str:
        return f'SFTModel/{self.run_name}'

    @property
    def checkpoint_relpath(self) -> str:
        return self.relpath

    @property
    def exists(self) -> bool:
        if not Project.config.CHECK_EXISTS_REMOTE:
            return False
        remote_path = os.path.join(cast(str, Project.config.GS_PATH), self.relpath)
        remote_files = G.get_remote_files(subfolder='SFTModel')
        found = any(f.startswith(remote_path) for f in remote_files)
        log.info(f"[SFTModel] {'✓ EXISTS' if found else '❌ NOT found'}: {self.run_name}")
        return found

    def get_requirements(self) -> Dict[str, Any]:
        return {
            'gres': f'gpu:{self.sft_gpus}',
            'nodes': 1,
            'cpus': self.sft_gpus * 2,
            'mem': '256GB',
            'requeue': True,
            'partition': 'flame',
            'account': 'aditirag',
            'qos': 'flame-32gpu_qos',
            'time': '2-00:00:00',
        }

    def construct(self, builder: Task):
        local_root = G.get_random_local_path()
        gs_root = cast(str, Project.config.GS_PATH)

        # 1. Download HFModel checkpoint locally
        src_ckpt_rel = self.hf_model.checkpoint_relpath
        local_model_path = os.path.join(local_root, src_ckpt_rel)
        builder.rsync_from_gs(
            os.path.join(gs_root, src_ckpt_rel),
            local_model_path,
            delete=False, checksum=False, skip_existing=False,
            check_exists=True, contents=True,
        )

        # 2. Prepare output directory
        save_folder = os.path.join(local_root, self.relpath)
        remote_folder = os.path.join(gs_root, self.relpath)
        builder.ensure_directory(save_folder)

        # 3. Build accelerate launch command
        open_instruct_path = os.path.expanduser('~/open-instruct')
        ds_config = 'configs/ds_configs/stage3_no_offloading_accelerate.conf'
        finetune_script = 'open_instruct/finetune.py'

        dataset_mixer_str = ' '.join(self.dataset_mixer)

        dataset_cache_dir = os.path.join(local_root, "dataset_cache")

        # Pre-cache dataset in single process to avoid NCCL barrier timeout.
        # After pre-caching, extract the config hash from the cache directory
        # and pass it to the multi-GPU run to guarantee it uses the same cache
        # (avoids hash mismatch due to dataset_commit_hash resolution timing).
        precache_parts = [
            'source ~/open-instruct/.venv/bin/activate &&',
            f'export HF_HOME={G.LOCAL_HF_PATH} &&',
            f'export TRITON_CACHE_DIR=/tmp/iwatts/triton && mkdir -p /tmp/iwatts/triton &&',
            f'cd {open_instruct_path} &&',
            f'python {finetune_script}',
            f'--model_name_or_path {local_model_path}',
            f'--dataset_mixer_list {dataset_mixer_str}',
            f'--max_seq_length {self.max_seq_length}',
            f'--dataset_local_cache_dir {dataset_cache_dir}',
            '--cache_dataset_only',
            '--no_push_to_hub',
            '--no_try_launch_beaker_eval_jobs',
        ]
        if self.add_bos:
            precache_parts.append('--add_bos')
        if self.chat_template_name:
            precache_parts.append(f'--chat_template_name {self.chat_template_name}')
        hash_file = os.path.join(local_root, "dataset_config_hash.txt")
        # After pre-cache completes, save the config hash from the cache directory
        precache_parts.append(
            f'&& ls -1 {dataset_cache_dir}/ | head -1 > {hash_file}'
            f' && echo "Saved dataset config hash: $(cat {hash_file})"'
        )
        builder.run_command(' '.join(precache_parts))
        exp_name = self.run_name[len(self.run_name)-64+1:] if len(self.run_name) > 64 else self.run_name

        cmd_parts = [
            'source ~/open-instruct/.venv/bin/activate &&',
            f'export HF_HOME={G.LOCAL_HF_PATH} &&',
            f'export TRITON_CACHE_DIR=/tmp/iwatts/triton && mkdir -p /tmp/iwatts/triton &&',
            f'cd {open_instruct_path} &&',
            'accelerate launch',
            '--mixed_precision bf16',
            f'--num_processes {self.sft_gpus}',
            '--num_machines 1',
            '--dynamo_backend no',
            '--use_deepspeed',
            f'--deepspeed_config_file {ds_config}',
            '--deepspeed_multinode_launcher standard',
            finetune_script,
            f'--exp_name {exp_name}', 
            f'--model_name_or_path {local_model_path}',
            f'--dataset_mixer_list {dataset_mixer_str}',
            '--use_flash_attn',
            f'--max_seq_length {self.max_seq_length}',
            f'--per_device_train_batch_size {self.per_device_train_batch_size}',
            f'--gradient_accumulation_steps {self.gradient_accumulation_steps}',
            f'--learning_rate {self.learning_rate}',
            f'--lr_scheduler_type {self.lr_scheduler_type}',
            f'--warmup_ratio {self.warmup_ratio}',
            f'--weight_decay {self.weight_decay}',
            f'--num_train_epochs {self.num_train_epochs}',
            '--report_to wandb',
            '--with_tracking',
            '--logging_steps 1',
            f'--wandb_project_name {cast(str, G.PROJECT_NAME)}',
            f'--wandb_entity {cast(str, G.WANDB_ENTITY)}',
            f'--output_dir {save_folder}',
            f'--seed {self.seed}',
            '--do_not_randomize_output_dir',
            '--no_push_to_hub',
            '--no_try_launch_beaker_eval_jobs',
            f'--dataset_local_cache_dir {dataset_cache_dir}',
            f'--dataset_config_hash $(cat {hash_file})',
            '--timeout 7200',
        ]

        if self.add_bos:
            cmd_parts.append('--add_bos')

        if self.chat_template_name:
            cmd_parts.append(f'--chat_template_name {self.chat_template_name}')

        builder.run_command(' '.join(cmd_parts))

        # 4. Upload to GCS
        builder.upload_to_gs(save_folder, remote_folder, directory=True)

        # Cleanup
        log.info(f"Cleaning up local directory: {local_root}")
        shutil.rmtree(local_root, ignore_errors=True)


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
