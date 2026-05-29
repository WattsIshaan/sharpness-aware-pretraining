"""OLMo model configuration utilities."""


def get_20m_base_model_config():
    """Return the base model configuration shared by pretrain and SFT."""
    return {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 8,
        'mlp_ratio': 8,
        'weight_tying': False,
        'alibi': False,
        'rope': True,
        'flash_attention': True,
        'attention_dropout': 0.0,
        'attention_layer_norm': False,
        'clip_qkv': None,
        'include_bias': False,
        'block_type': 'sequential',
        'layer_norm_type': 'rms',
        'layer_norm_with_affine': True,
        'layer_norm_eps': 1e-6,
        'bias_for_layer_norm': False,
        'attention_layer_norm_with_affine': False,
        'activation_type': 'swiglu',
        'residual_dropout': 0.0,
        'embedding_dropout': 0.0,
        'max_sequence_length': 1024,
        'vocab_size': 50280,
        'embedding_size': 50304,
        'eos_token_id': 0,
        'pad_token_id': 1,
        'init_device': 'cuda',
        'init_fn': 'normal',
        'init_std': 0.02,
        'init_cutoff_factor': 3,
    }

def get_60m_base_model_config():
    """Return the base model configuration shared by pretrain and SFT."""
    return {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 8,
        'mlp_ratio': 8,
        'weight_tying': False,
        'alibi': False,
        'rope': True,
        'flash_attention': True,
        'attention_dropout': 0.0,
        'attention_layer_norm': False,
        'clip_qkv': None,
        'include_bias': False,
        'block_type': 'sequential',
        'layer_norm_type': 'rms',
        'layer_norm_with_affine': True,
        'layer_norm_eps': 1e-6,
        'bias_for_layer_norm': False,
        'attention_layer_norm_with_affine': False,
        'activation_type': 'swiglu',
        'residual_dropout': 0.0,
        'embedding_dropout': 0.0,
        'max_sequence_length': 1024,
        'vocab_size': 50280,
        'embedding_size': 50304,
        'eos_token_id': 0,
        'pad_token_id': 1,
        'init_device': 'cuda',
        'init_fn': 'normal',
        'init_std': 0.02,
        'init_cutoff_factor': 3,
    }

def get_150m_base_model_config():
    """Return the base model configuration shared by pretrain and SFT."""
    return {
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'mlp_ratio': 8,
        'weight_tying': False,
        'alibi': False,
        'rope': True,
        'flash_attention': True,
        'attention_dropout': 0.0,
        'attention_layer_norm': False,
        'clip_qkv': None,
        'include_bias': False,
        'block_type': 'sequential',
        'layer_norm_type': 'rms',
        'layer_norm_with_affine': True,
        'layer_norm_eps': 1e-6,
        'bias_for_layer_norm': False,
        'attention_layer_norm_with_affine': False,
        'activation_type': 'swiglu',
        'residual_dropout': 0.0,
        'embedding_dropout': 0.0,
        'max_sequence_length': 1024,
        'vocab_size': 50280,
        'embedding_size': 50304,
        'eos_token_id': 0,
        'pad_token_id': 1,
        'init_device': 'cuda',
        'init_fn': 'normal',
        'init_std': 0.02,
        'init_cutoff_factor': 3,
    }

def get_300m_base_model_config():
    """Return the base model configuration shared by pretrain and SFT."""
    return {
        'd_model': 1024,
        'n_heads': 16,
        'n_layers': 16,
        'mlp_ratio': 8,
        'weight_tying': False,
        'alibi': False,
        'rope': True,
        'flash_attention': True,
        'attention_dropout': 0.0,
        'attention_layer_norm': False,
        'clip_qkv': None,
        'include_bias': False,
        'block_type': 'sequential',
        'layer_norm_type': 'rms',
        'layer_norm_with_affine': True,
        'layer_norm_eps': 1e-6,
        'bias_for_layer_norm': False,
        'attention_layer_norm_with_affine': False,
        'activation_type': 'swiglu',
        'residual_dropout': 0.0,
        'embedding_dropout': 0.0,
        'max_sequence_length': 1024,
        'vocab_size': 50280,
        'embedding_size': 50304,
        'eos_token_id': 0,
        'pad_token_id': 1,
        'init_device': 'cuda',
        'init_fn': 'normal',
        'init_std': 0.02,
        'init_cutoff_factor': 3,
    }

def get_1b_midtraining_config():
    return {
        'd_model': 2048,
        'n_heads': 16,
        'n_layers': 16,
        'mlp_ratio': 8,
        'weight_tying': False,
        'alibi': False,
        'rope': True,
        'rope_theta': 500000,
        'flash_attention': True,
        'attention_dropout': 0.0,
        'include_bias': False,
        'block_type': 'sequential',
        'layer_norm_type': 'rms',
        'layer_norm_with_affine': True,
        'layer_norm_eps': 1e-6,
        'bias_for_layer_norm': False,
        'attention_layer_norm': True,
        'attention_layer_norm_with_affine': True,
        'norm_after': True,
        'activation_type': 'swiglu',
        'residual_dropout': 0.0,
        'embedding_dropout': 0.0,
        'max_sequence_length': 2048,
        'vocab_size': 100278,
        'embedding_size': 100352,   
        'eos_token_id': 100257,
        'pad_token_id': 100277,
        'init_device': 'cuda',
        'init_fn': 'normal',
        'init_std': 0.02,
        'init_cutoff_factor': 3,
    }


def get_train_config(
    run_name,
    save_folder,
    train_data_paths,
    eval_datasets=None,
    model_size = '20m',
    train_data_label_mask_paths=None,
    optimizer='adamw',
    learning_rate=3.0e-4,
    weight_decay=0.1,
    max_duration: int | str = '1ep',
    stop_at=None,
    seed=6198,
    scheduler_name='cosine_with_warmup',
    scheduler_t_warmup=5000,
    scheduler_alpha_f=0.1,
    scheduler_anneal_begin=None,
    optimizer_betas=(0.9, 0.95),
    momentum=0.9,
    sam_rho=0.05,
    sam_base_optimizer='adamw',
    optimizer_eps=1e-8,
    decay_embeddings=True,
    global_train_batch_size=256,
    device_train_microbatch_size=32,
    device_eval_batch_size=None,
    eval_interval=5000,
    save_interval_unsharded=5000,
    save_num_unsharded_checkpoints_to_keep=-1,
    try_load_latest_save=True,
    max_grad_norm=1.0,
    load_path=None,
    ewc_lambda=None,
    ewc_fisher_batches=100,
    ewc_fisher_paths=None,
    ewc_fisher_label_mask_paths=None,
    restore_dataloader=False,
    reset_optimizer_state=False,
    model_overrides=None,
    wandb_project=None,
    wandb_entity=None,
    wandb_id=None,
    wandb_resume='allow',
    run_sync_cmd=None,
    dtype="uint16",
    distributed_strategy='ddp',
    **overrides
):
    """
    Create a unified training configuration for OLMo pretraining.
    
    Args:
        run_name: Name of the training run
        save_folder: Path to save checkpoints
        train_data_paths: List of paths to training data files
        train_data_label_mask_paths: List of paths to training data label mask files
        eval_datasets: Dict of evaluation datasets {name: [paths]}
        optimizer: Optimizer name (e.g., 'adamw', 'lionw')
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay coefficient
        max_duration: Training duration (e.g., '1ep' for 1 epoch, '4e9T' for 4B tokens)
        stop_at: Optional step to stop at
        seed: Random seed
        scheduler_name: Scheduler type ('cosine_with_warmup', 'linear_with_warmup', etc.)
        scheduler_t_warmup: Warmup steps
        scheduler_alpha_f: Final learning rate multiplier
        optimizer_betas: Optimizer beta parameters
        optimizer_eps: Optimizer epsilon
        decay_embeddings: Whether to apply weight decay to embeddings (False for midtraining)
        global_train_batch_size: Global batch size
        device_train_microbatch_size: Microbatch size for training
        device_eval_batch_size: Batch size for evaluation (defaults to device_train_microbatch_size)
        eval_interval: Steps between evaluations
        save_interval_unsharded: Steps between unsharded checkpoints
        save_num_unsharded_checkpoints_to_keep: Number of checkpoints to keep (-1 for all)
        max_grad_norm: Maximum gradient norm for clipping
        load_path: Path to checkpoint to load (for SFT)
        ewc_lambda: If set, enable Elastic Weight Consolidation (``scripts/train_ewc.py``)
        ewc_fisher_batches: Batches used to estimate diagonal Fisher for EWC
        ewc_fisher_paths: Memmap paths for Fisher (e.g. pretrain val); default SFT train if unset
        ewc_fisher_label_mask_paths: Optional masks paired with ``ewc_fisher_paths``
        restore_dataloader: Whether to restore dataloader state
        reset_optimizer_state: Whether to reset optimizer state (for SFT)
        model_overrides: Dict of model config overrides
        wandb_project: Weights & Biases project name (optional)
        wandb_entity: Weights & Biases entity name (optional)
        wandb_id: Weights & Biases run ID (optional)
        wandb_resume: Weights & Biases resume mode ('allow', 'never', 'must')
        **overrides: Additional top-level config overrides
    """
    if device_eval_batch_size is None:
        device_eval_batch_size = device_train_microbatch_size
    
    if model_size == '20m':
        model_config = get_20m_base_model_config()
    elif model_size == '60m':
        model_config = get_60m_base_model_config()
    elif model_size == '150m':
        model_config = get_150m_base_model_config()
    elif model_size == '300m':
        model_config = get_300m_base_model_config()
    elif model_size == '1b':
        model_config = get_1b_midtraining_config()
    else:
        raise ValueError("Invalid Model Size")

    if model_overrides:
        model_config.update(model_overrides)

    # if max_duration.endswith('T') and scheduler_name == 'cosine_with_warmup':
    #     max_tokens = int(float(max_duration[:-1].strip()))
    #     scheduler_t_warmup = int(0.1 * (max_tokens // (global_train_batch_size * model_config["max_sequence_length"])))
    
    config = {
        'run_name': run_name,
        'seed': seed,
        'dry_run': False,
        
        'model': model_config,
        
        'ddp': {
            'grad_sync_mode': 'batch',
            'find_unused_params': False,
        },
        
        'compile': {
            'mode': 'default',
        },
        
        'optimizer': {
            'name': optimizer,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'eps': optimizer_eps,
            'decay_norm_and_bias': True,
            'decay_embeddings': decay_embeddings,
            'betas': list(optimizer_betas),
            'metrics_log_interval': 10,
            'momentum': momentum,
            'sam_rho': sam_rho,
            'sam_base_optimizer': sam_base_optimizer,
        },
        
        'scheduler': {
            'name': scheduler_name,
            't_warmup': scheduler_t_warmup,
            'alpha_f': scheduler_alpha_f,
            'warmup_min_lr': 0,
            'anneal_begin': scheduler_anneal_begin,
        },
        
        'tokenizer': {
            'identifier': 'tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json',
            'truncate_direction': 'right',
        },
        
        'save_folder': save_folder,
        'save_overwrite': True,
        'save_interval_unsharded': save_interval_unsharded,
        'save_num_unsharded_checkpoints_to_keep': save_num_unsharded_checkpoints_to_keep,
        'load_path': load_path,
        'try_load_latest_save': try_load_latest_save,
        'run_sync_cmd': run_sync_cmd,
        
        'max_duration': max_duration,
        'stop_at': stop_at,
        'global_train_batch_size': global_train_batch_size,
        'device_train_microbatch_size': device_train_microbatch_size,
        
        'precision': 'amp_bf16',
        'distributed_strategy': distributed_strategy,
        'gen1_gc_interval': 1,
        'max_grad_norm': max_grad_norm,
        'max_grad_norm_ratio': None,
        
        'speed_monitor': {'window_size': 20},
        
        'eval_interval': eval_interval,
        'eval_subset_num_batches': -1,
        'device_eval_batch_size': device_eval_batch_size,
        
        'data': {
            'pad_direction': 'right',
            'num_workers': 4,
            'drop_last': True,
            'pin_memory': True,
            'prefetch_factor': 8,
            'persistent_workers': True,
            'timeout': 0,
            'generate_attention_mask': True,
            'paths': train_data_paths,
            'memmap_dtype': dtype
        },
    }
    
    # Add wandb config if provided
    if wandb_project:
        config['wandb'] = {
            'name': run_name,
            'project': wandb_project,
            'entity': wandb_entity,
            # 'id': wandb_id,
            'resume': wandb_resume,
        }
    
    # Add evaluators if eval datasets are provided
    if eval_datasets:
        config['evaluators'] = []
        if "pretrain" in eval_datasets:
            config['evaluators'].append({
                'label': 'pretrain-perplexity',
                'data': {
                    'num_workers': 0,
                    'drop_last': True,
                    'datasets': eval_datasets["pretrain"],
                    'generate_attention_mask': True,
                    'memmap_dtype': dtype
                },
            })

        if "sft" in eval_datasets:
            for k, v in eval_datasets["sft"].items():
                config['evaluators'].append({
                    'label': k,
                    'data': {
                        'num_workers': 0,
                        'drop_last': True,
                        'generate_attention_mask': True,
                        'paths': v["data_paths"],
                        'label_mask_paths': v["mask_paths"],
                        'memmap_dtype': dtype
                    },
                })
        if "downstream" in eval_datasets:
            for k in eval_datasets["downstream"]:
                config['evaluators'].append({
                    'label': k,
                    'type': 'downstream',
                })

    # Add label mask paths if provided
    if train_data_label_mask_paths:
        config['data']['label_mask_paths'] = train_data_label_mask_paths
    
    # Handle SFT-specific settings
    if load_path is not None:
        config['restore_dataloader'] = restore_dataloader
        config['no_pre_train_checkpoint'] = True
        config['reset_optimizer_state'] = reset_optimizer_state

    if ewc_lambda is not None:
        config['ewc_lambda'] = ewc_lambda
        config['ewc_fisher_batches'] = ewc_fisher_batches
    if ewc_fisher_paths is not None:
        config['ewc_fisher_paths'] = ewc_fisher_paths
    if ewc_fisher_label_mask_paths is not None:
        config['ewc_fisher_label_mask_paths'] = ewc_fisher_label_mask_paths
    
    # Apply any additional overrides
    config.update(overrides)
    return config
