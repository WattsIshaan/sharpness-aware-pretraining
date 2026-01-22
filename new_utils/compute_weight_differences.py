"""Compute L2 weight differences between CPT and base models."""

import os
import csv
import argparse
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil

import torch
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def download_checkpoint_from_gs(gs_path: str, local_dir: str) -> str:
    """Download checkpoint from Google Cloud Storage.
    
    Args:
        gs_path: Full GCS path to checkpoint directory
        local_dir: Local directory to download to
        
    Returns:
        Path to downloaded checkpoint directory
    """
    log.info(f"Downloading checkpoint from {gs_path} to {local_dir}")
    result = subprocess.run(
        ['gsutil', '-m', 'cp', '-r', gs_path, local_dir],
        capture_output=True,
        text=True,
        timeout=3600
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download from {gs_path}: {result.stderr}")
    
    # Return the local path where files were downloaded
    checkpoint_name = os.path.basename(gs_path.rstrip('/'))
    return os.path.join(local_dir, checkpoint_name)


def load_model_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load model state dict from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory containing model.pt or similar
        
    Returns:
        Dictionary of parameter name to tensor
    """
    # Try different common checkpoint file names
    possible_files = [
        'model.pt',
        'pytorch_model.bin',
        'model.safetensors',
        'checkpoint.pt',
    ]
    
    checkpoint_dir = Path(checkpoint_path)
    state_dict = None
    
    for fname in possible_files:
        fpath = checkpoint_dir / fname
        if fpath.exists():
            log.info(f"Loading weights from {fpath}")
            if fname.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(str(fpath))
                except ImportError:
                    log.warning("safetensors not installed, skipping .safetensors file")
                    continue
            else:
                checkpoint = torch.load(fpath, map_location='cpu', weights_only=False)
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
            break
    
    if state_dict is None:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_path}")
    
    return state_dict


def compute_l2_difference(base_state_dict: Dict[str, torch.Tensor], 
                         cpt_state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute per-layer normalized L2 distance with aggregation.
    
    For each layer l: D_l = ||θ1^(l) - θ0^(l)||_2 / ||θ0^(l)||_2
    
    Then aggregate:
    - Unweighted mean: (1/L) * Σ D_l
    - Weighted mean: Σ(n_l * D_l) / Σ n_l  (weighted by parameter count) ← BETTER
    
    This approach:
    - Respects model structure
    - Prevents embeddings/LayerNorm from dominating
    - Stable across seeds
    - Interpretable ("fractional change per layer")
    
    Args:
        base_state_dict: Base model state dict
        cpt_state_dict: CPT model state dict
        
    Returns:
        Dictionary with metrics
    """
    per_layer_normalized = []
    per_layer_param_counts = []
    
    # Get common keys
    common_keys = set(base_state_dict.keys()) & set(cpt_state_dict.keys())
    
    if not common_keys:
        log.warning("No common keys found between base and CPT models!")
        return {
            'mean_normalized_l2': 0.0,
            'weighted_mean_normalized_l2': 0.0,
            'num_layers': 0,
            'total_params': 0
        }
    
    log.info(f"Computing per-layer normalized L2 over {len(common_keys)} layers")
    
    for key in sorted(common_keys):
        base_param = base_state_dict[key]
        cpt_param = cpt_state_dict[key]
        
        # Skip if shapes don't match
        if base_param.shape != cpt_param.shape:
            log.warning(f"Shape mismatch for {key}: {base_param.shape} vs {cpt_param.shape}")
            continue
        
        # Compute L2 norm of difference and base for this layer
        diff = cpt_param - base_param
        layer_diff_norm = torch.norm(diff.float(), p=2).item()
        layer_base_norm = torch.norm(base_param.float(), p=2).item()
        
        # Per-layer normalized L2: D_l = ||Δθ^(l)||_2 / ||θ0^(l)||_2
        if layer_base_norm > 0:
            normalized_l2 = layer_diff_norm / layer_base_norm
            per_layer_normalized.append(normalized_l2)
            per_layer_param_counts.append(base_param.numel())
    
    # Compute aggregated metrics
    if not per_layer_normalized:
        return {
            'mean_normalized_l2': 0.0,
            'weighted_mean_normalized_l2': 0.0,
            'num_layers': 0,
            'total_params': 0
        }
    
    # Unweighted mean: (1/L) * Σ D_l
    mean_normalized_l2 = sum(per_layer_normalized) / len(per_layer_normalized)
    
    # Weighted mean by parameter count: Σ(n_l * D_l) / Σ n_l (BETTER)
    weighted_sum = sum(n * d for n, d in zip(per_layer_param_counts, per_layer_normalized))
    total_params = sum(per_layer_param_counts)
    weighted_mean_normalized_l2 = weighted_sum / total_params if total_params > 0 else 0.0
    
    return {
        'mean_normalized_l2': mean_normalized_l2,
        'weighted_mean_normalized_l2': weighted_mean_normalized_l2,  # PRIMARY METRIC (better)
        'num_layers': len(per_layer_normalized),
        'total_params': total_params
    }


def extract_token_budget_from_path(path: str) -> Optional[str]:
    """Extract token budget from path or model name.
    
    Expected patterns:
        - tk16B or tk4B in the path -> returns "16" or "4"
        - 16B or 4B in the path -> returns "16" or "4"
        - 4000M -> returns "4"
    
    Args:
        path: Path or model name
        
    Returns:
        Token budget in billions as string (e.g., '4', '16'), or None if not found
    """
    # First try to find tkNB pattern (preferred)
    match = re.search(r'tk(\d+)B', path)
    if match:
        return match.group(1)
    
    # Fallback: look for any NB pattern
    match = re.search(r'(\d+)B', path)
    if match:
        return match.group(1)
    
    # Check for NM pattern (e.g., 4000M = 4B)
    match = re.search(r'(\d+)M', path)
    if match:
        millions = int(match.group(1))
        # Convert millions to billions if it makes sense
        if millions >= 1000:
            return str(millions // 1000)
    
    return None


def parse_base_model_info(base_model_path: str, gs_path: str) -> Dict[str, Any]:
    """Parse metadata from base model path including token budget.
    
    Expected format: gs://path/PretrainedModel/{model_name}/final-unsharded
    or gs://path/tk4B/PretrainedModel/{model_name}/final-unsharded
    
    Args:
        base_model_path: Full GCS path or model name
        gs_path: Root GCS path for project
        
    Returns:
        Dictionary with parsed metadata including token_budget
    """
    info = {
        'model_name': None,
        'token_budget': None,
        'checkpoint_path': None,
    }
    
    # Determine the checkpoint path and model name
    if base_model_path.startswith('gs://'):
        info['checkpoint_path'] = base_model_path
        path_parts = base_model_path.rstrip('/').split('/')
        if 'PretrainedModel' in path_parts:
            idx = path_parts.index('PretrainedModel')
            info['model_name'] = path_parts[idx + 1] if idx + 1 < len(path_parts) else path_parts[-1]
        else:
            info['model_name'] = path_parts[-1]
    else:
        info['model_name'] = base_model_path
        info['checkpoint_path'] = f"{gs_path}/PretrainedModel/{base_model_path}/final-unsharded"
    
    # Extract token budget from the full path
    info['token_budget'] = extract_token_budget_from_path(base_model_path)
    
    return info


def extract_cpt_learning_rate(run_name: str) -> Optional[str]:
    """Extract the CPT learning rate from run name.
    
    For muon models, gets the LAST muon_lr occurrence (CPT muon LR).
    For other models, gets the second lr occurrence (CPT LR).
    
    Args:
        run_name: The run name string
        
    Returns:
        Learning rate as formatted string (e.g., '1.00e-04'), or None
    """
    # Pattern for scientific notation in learning rates
    # Matches patterns like: 1e-4, 1.5e-4, 1_5e-4, etc.
    sci_pattern = r'(\d+(?:[._]\d+)?[eE][+-]?\d+)'
    
    # For muon models, look specifically for muon_lr and get the LAST one
    muon_lr_matches = re.findall(r'muon_lr' + sci_pattern, run_name, re.IGNORECASE)
    if muon_lr_matches:
        # Use the LAST muon_lr (CPT muon LR)
        lr_str = muon_lr_matches[-1].replace('_', '.')
        try:
            return f"{float(lr_str):.2e}"
        except ValueError:
            return muon_lr_matches[-1]
    
    # For non-muon models, find all lr patterns
    lr_matches = re.findall(r'lr' + sci_pattern, run_name, re.IGNORECASE)
    
    if len(lr_matches) >= 2:
        # Use the second learning rate (CPT LR)
        lr_str = lr_matches[1].replace('_', '.')
        try:
            return f"{float(lr_str):.2e}"
        except ValueError:
            return lr_matches[1]
    elif len(lr_matches) == 1:
        # Only one LR found, use it
        lr_str = lr_matches[0].replace('_', '.')
        try:
            return f"{float(lr_str):.2e}"
        except ValueError:
            return lr_matches[0]
    
    return None


def parse_cpt_model_info(cpt_path: str) -> Dict[str, Any]:
    """Parse metadata from CPT model path.
    
    Expected format: CPTModel/{dataset}/{run_name}/final-unsharded
    or gs://path/tk4B/CPTModel/{dataset}/{run_name}/final-unsharded
    Where run_name contains: {base_model_name}_tok{tokens}_opt{optimizer}_lr{lr}_wd{wd}_...
    
    Args:
        cpt_path: GCS path to CPT model
        
    Returns:
        Dictionary with parsed metadata
    """
    parts = cpt_path.split('/')
    
    info = {
        'cpt_path': cpt_path,
        'dataset': None,
        'run_name': None,
        'base_model_name': None,
        'base_token_budget': None,  # Token budget of the base model (in billions)
        'train_tokens': None,
        'optimizer': None,
        'learning_rate': None,
        'weight_decay': None,
        'batch_size': None,
    }
    
    # Extract token budget from the full path (e.g., tk4B, tk16B)
    info['base_token_budget'] = extract_token_budget_from_path(cpt_path)
    
    # Find CPTModel in path
    try:
        cpt_idx = parts.index('CPTModel')
        if len(parts) > cpt_idx + 2:
            info['dataset'] = parts[cpt_idx + 1]
            info['run_name'] = parts[cpt_idx + 2]
            
            # Parse run_name for metadata
            run_name = info['run_name']
            
            # Extract CPT learning rate (second LR occurrence)
            info['learning_rate'] = extract_cpt_learning_rate(run_name)
            
            tokens = run_name.split('_tok')
            if len(tokens) > 1:
                # Extract base model name (everything before _tok)
                info['base_model_name'] = tokens[0]
                
                # Parse the rest
                params_str = '_tok' + tokens[1]
                param_parts = params_str.split('_')
                
                for i, part in enumerate(param_parts):
                    if part.startswith('tok'):
                        info['train_tokens'] = part[3:]
                    elif part.startswith('opt'):
                        info['optimizer'] = part[3:]
                    elif part.startswith('wd'):
                        info['weight_decay'] = part[2:]
                    elif part.startswith('bs'):
                        info['batch_size'] = part[2:]
                    # Note: learning_rate is extracted separately above to get the second occurrence
    except (ValueError, IndexError) as e:
        log.warning(f"Failed to parse CPT path {cpt_path}: {e}")
    
    return info

def cpt_adamw(filename: str) -> Optional[str]:
    """Extract learning rate from filename."""
    # Pattern for scientific notation
    sci_pattern = r'(\d+(?:[._]\d+)?[eE][+-]?\d+)'
    
    # Try to find CPT lr (second occurrence)
    lrs = re.findall(r'OLMo2-20m', filename)

    return len(lrs) < 1


def compute_weight_differences_for_dataset(
    cpt_dataset: str,
    base_model_paths: List[str],
    gs_path: str,
    output_csv: str,
    cpt_optimizer: str = None,
    cpt_learning_rate: float = None
):
    """Compute per-layer normalized L2 weight differences with aggregation.
    
    Method (per-layer normalization):
    1. For each layer l: D_l = ||θ_cpt^(l) - θ_base^(l)||_2 / ||θ_base^(l)||_2
    2. Aggregate with weighted mean: Σ(n_l * D_l) / Σ n_l
    
    Primary metric: weighted_mean_normalized_l2 (weighted by parameter count)
    
    Why this works:
    - Respects model structure
    - Prevents embeddings/LayerNorm from dominating
    - Stable across seeds
    - Interpretable ("fractional change per layer")
    - This is what people actually trust internally
    
    Args:
        cpt_dataset: Name of CPT dataset (e.g., 'tulu', 'alpaca', 'gsm8k')
        base_model_paths: List of base model GCS paths
        gs_path: Root GCS path for project
        output_csv: Path to output CSV file
        cpt_optimizer: Optional filter for optimizer
        cpt_learning_rate: Optional filter for learning rate
    """
    # List all CPT models for this dataset
    cpt_base_path = f"{gs_path}/CPTModel/{cpt_dataset}"
    log.info(f"Listing CPT models from {cpt_base_path}")
    
    result = subprocess.run(
        ['gsutil', 'ls', '-d', f"{cpt_base_path}/*/"],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        log.error(f"Failed to list CPT models: {result.stderr}")
        return
    
    cpt_model_dirs = [line.strip().rstrip('/') for line in result.stdout.split('\n') if line.strip()]
    log.info(f"Found {len(cpt_model_dirs)} CPT models")
    
    results = []
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download and cache base models, organized by token budget
        base_model_cache = {}  # {model_name: state_dict}
        base_model_by_tokens = {}  # {token_budget: {model_name: state_dict}}
        
        for base_path in base_model_paths:
            # Parse base model information
            base_info = parse_base_model_info(base_path, gs_path)
            base_model_name = base_info['model_name']
            base_checkpoint_path = base_info['checkpoint_path']
            token_budget = base_info['token_budget']

            
            log.info(f"Processing base model: {base_model_name} (token budget: {token_budget})")
            
            try:
                local_base_path = download_checkpoint_from_gs(base_checkpoint_path, temp_dir)
                base_state_dict = load_model_state_dict(local_base_path)
                
                # Cache by model name
                base_model_cache[base_model_name] = {
                    'state_dict': base_state_dict,
                    'token_budget': token_budget
                }
                
                # Also cache by token budget for easier lookup
                if token_budget:
                    if token_budget not in base_model_by_tokens:
                        base_model_by_tokens[token_budget] = {}
                    base_model_by_tokens[token_budget][base_model_name] = base_state_dict
                
                log.info(f"Loaded base model {base_model_name} with {len(base_state_dict)} parameters")
            except Exception as e:
                log.error(f"Failed to load base model {base_path}: {e}")
                continue
        
        # Process each CPT model
        for cpt_dir in cpt_model_dirs:
            cpt_checkpoint_path = f"{cpt_dir}/final-unsharded"
            
            # Parse metadata from path
            info = parse_cpt_model_info(cpt_dir)
            
            # Apply filters if specified
            if cpt_optimizer and info['optimizer'] != cpt_optimizer:
                continue
            if cpt_learning_rate and info['learning_rate'] != str(cpt_learning_rate):
                continue
            
            log.info(f"Processing CPT model: {info['run_name']}")
            log.info(f"  Base model: {info['base_model_name']}, Token budget: {info['base_token_budget']}")

            if cpt_adamw(info['run_name']):
                continue
            
            try:
                # Download CPT model
                local_cpt_path = download_checkpoint_from_gs(cpt_checkpoint_path, temp_dir)
                cpt_state_dict = load_model_state_dict(local_cpt_path)
                
                # Find matching base model by name AND token budget
                base_name = info['base_model_name']
                base_token_budget = info['base_token_budget']
                
                # First try to match by exact name
                if base_name in base_model_cache:
                    cached_base = base_model_cache[base_name]
                    base_state_dict = cached_base['state_dict']
                    matched_token_budget = cached_base['token_budget']
                    
                    # Check if token budgets match
                    if base_token_budget and matched_token_budget and base_token_budget != matched_token_budget:
                        log.warning(f"Token budget mismatch: CPT expects {base_token_budget}, but base has {matched_token_budget}")
                        log.warning(f"Skipping CPT model {info['run_name']}")
                        continue
                    
                    log.info(f"Matched with base model: {base_name} (token budget: {matched_token_budget})")
                
                # If not found by name, try to match by token budget
                elif base_token_budget and base_token_budget in base_model_by_tokens:
                    log.info(f"Base model {base_name} not found, trying to match by token budget {base_token_budget}")
                    models_with_budget = base_model_by_tokens[base_token_budget]
                    
                    if len(models_with_budget) == 1:
                        # Only one model with this token budget, use it
                        matched_name = list(models_with_budget.keys())[0]
                        base_state_dict = models_with_budget[matched_name]
                        log.info(f"Matched with base model: {matched_name} (token budget: {base_token_budget})")
                    else:
                        log.warning(f"Multiple base models found with token budget {base_token_budget}: {list(models_with_budget.keys())}")
                        log.warning(f"Skipping CPT model {info['run_name']} - ambiguous match")
                        continue
                else:
                    log.warning(f"No matching base model found for {base_name} (token budget: {base_token_budget}), skipping")
                    continue
                
                # Compute per-layer normalized L2 with aggregation (PRIMARY METRIC)
                l2_metrics = compute_l2_difference(base_state_dict, cpt_state_dict)
                
                # Store result - ONE ROW PER MODEL with aggregated metrics only
                result_entry = {
                    'cpt_dataset': cpt_dataset,
                    'base_model_name': info['base_model_name'],
                    'base_token_budget': info['base_token_budget'],
                    'run_name': info['run_name'],
                    'train_tokens': info['train_tokens'],
                    'optimizer': info['optimizer'],
                    'learning_rate': info['learning_rate'],
                    'weight_decay': info['weight_decay'],
                    'batch_size': info['batch_size'],
                    'weighted_mean_normalized_l2': l2_metrics['weighted_mean_normalized_l2'],  # PRIMARY METRIC
                    'mean_normalized_l2': l2_metrics['mean_normalized_l2'],
                    'num_layers': l2_metrics['num_layers'],
                    'total_params': l2_metrics['total_params'],
                    'cpt_path': cpt_checkpoint_path,
                }
                
                results.append(result_entry)
                log.info(f"Weighted mean normalized L2: {l2_metrics['weighted_mean_normalized_l2']:.4f} (unweighted: {l2_metrics['mean_normalized_l2']:.4f})")
                
                # Clean up CPT model to save space
                shutil.rmtree(local_cpt_path, ignore_errors=True)
                
            except Exception as e:
                log.error(f"Failed to process {cpt_dir}: {e}")
                continue
    
    # Write results to CSV
    if results:
        log.info(f"Writing {len(results)} results to {output_csv}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Fixed columns - one row per model with aggregated metrics only
        fieldnames = [
            'cpt_dataset',
            'base_model_name',
            'base_token_budget',
            'run_name',
            'train_tokens',
            'optimizer',
            'learning_rate',
            'weight_decay',
            'batch_size',
            'weighted_mean_normalized_l2',  # PRIMARY METRIC (weighted by param count)
            'mean_normalized_l2',  # Unweighted mean
            'num_layers',
            'total_params',
            'cpt_path',
        ]
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        
        log.info(f"Results written to {output_csv}")
        log.info(f"  Total rows: {len(results)}, columns: {len(fieldnames)}")
    else:
        log.warning("No results to write")


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-layer normalized L2 weight differences with aggregation.\n\n'
                    'Method:\n'
                    '  1. For each layer: D_l = ||Δθ^(l)||_2 / ||θ_base^(l)||_2\n'
                    '  2. Aggregate with weighted mean by parameter count (better)\n\n'
                    'Primary metric: weighted_mean_normalized_l2\n'
                    'This respects model structure and prevents embeddings from dominating.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--cpt-dataset',
        type=str,
        required=True,
        help='CPT dataset name (e.g., tulu, alpaca, gsm8k)'
    )
    parser.add_argument(
        '--base-models',
        type=str,
        nargs='+',
        required=True,
        help='List of base model paths or names'
    )
    parser.add_argument(
        '--gs-path',
        type=str,
        required=True,
        help='Root GCS path (e.g., gs://bucket-name/project)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default=None,
        help='Filter by optimizer (optional)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Filter by learning rate (optional)'
    )
    
    args = parser.parse_args()
    
    compute_weight_differences_for_dataset(
        cpt_dataset=args.cpt_dataset,
        base_model_paths=args.base_models,
        gs_path=args.gs_path,
        output_csv=args.output_csv,
        cpt_optimizer=args.optimizer,
        cpt_learning_rate=args.learning_rate
    )


if __name__ == '__main__':
    main()
