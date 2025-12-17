import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from olmo.model import OLMo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def evaluate_file(model: OLMo, data_path: str, chunk_size: int, device: torch.device, mask_path: str = None) -> float:
    """Evaluate model on a single data file."""
    # Load data using memmap
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    
    # Load mask if provided
    mask = None
    if mask_path:
        # Try loading mask with different dtypes
        import os
        file_size = os.path.getsize(mask_path)
        data_len = len(data)
        
        # Determine dtype based on file size
        if file_size == data_len * 2:
            # uint16 (2 bytes per element)
            mask = np.memmap(mask_path, dtype=np.uint16, mode='r')
        elif file_size == data_len:
            # uint8 or bool (1 byte per element)
            mask = np.memmap(mask_path, dtype=np.uint8, mode='r')
        else:
            log.warning(f"Mask file size ({file_size} bytes) doesn't match expected size for data length {data_len}. Expected {data_len} or {data_len*2} bytes. Ignoring mask.")
            mask = None
        
        if mask is not None and len(mask) != len(data):
            log.warning(f"Mask length ({len(mask)}) does not match data length ({len(data)}). Ignoring mask.")
            mask = None
    
    total_loss = 0.0
    total_tokens = 0
    
    # Process data in chunks
    num_chunks = len(data) // chunk_size
    
    log.info(f"Processing {num_chunks} chunks from {data_path}" + (" with mask" if mask is not None else ""))
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = data[start_idx:end_idx]
        
        # Convert to tensor and add batch dimension
        input_ids = torch.from_numpy(np.array(chunk)).long().unsqueeze(0).to(device)
        
        # Load mask chunk if available
        mask_chunk = None
        if mask is not None:
            mask_chunk = torch.from_numpy(np.array(mask[start_idx:end_idx])).bool().unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get model output
            output = model(input_ids)
            logits = output.logits
            
            # Compute loss (predict next token)
            # logits shape: (batch_size, seq_len, vocab_size)
            # We use logits[:-1] to predict labels[1:]
            logits_for_loss = logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            
            # Flatten for cross entropy
            logits_flat = logits_for_loss.view(-1, logits_for_loss.size(-1))
            labels_flat = labels.view(-1)
            
            # Compute loss without reduction if we have a mask
            if mask_chunk is not None:
                # Shift mask to align with labels (we're predicting next token)
                mask_for_labels = mask_chunk[:, 1:].contiguous()
                mask_flat = mask_for_labels.view(-1)
                
                # Compute per-token loss
                loss_per_token = F.cross_entropy(logits_flat, labels_flat, reduction='none')
                
                # Apply mask (only count tokens where mask is True/1)
                masked_loss = loss_per_token * mask_flat.float()
                
                total_loss += masked_loss.sum().item()
                total_tokens += mask_flat.sum().item()
            else:
                # No mask - compute loss on all tokens
                loss = F.cross_entropy(logits_flat, labels_flat, reduction='sum')
            
                total_loss += loss.item()
                total_tokens += labels_flat.numel()
    
    # Return average loss per token
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Evaluate OLMo model on tokenized data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (e.g., cpu, cuda, cuda:0)')
    parser.add_argument('--data_path', type=str, required=True, help='Comma-separated paths to data files (uint16 format)')
    parser.add_argument('--mask_path', type=str, default=None, help='Comma-separated paths to mask files (uint16 format, 0=ignore, 1=include). Must match order of data_path.')
    parser.add_argument('--chunk_size', type=int, required=True, help='Size of chunks to process')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output JSON file')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    log.info(f"Using device: {device}")
    
    # Load model
    log.info(f"Loading model from {args.model_path}")
    model = OLMo.from_checkpoint(args.model_path, device=args.device)
    model.eval()
    
    # Parse comma-separated data paths
    data_paths = [p.strip() for p in args.data_path.split(',') if p.strip()]
    
    # Parse comma-separated mask paths (optional)
    # Keep empty strings to maintain alignment, but convert to None
    mask_paths = []
    if args.mask_path:
        raw_mask_paths = [p.strip() for p in args.mask_path.split(',')]
        # Convert empty strings to None for proper alignment
        mask_paths = [p if p else None for p in raw_mask_paths]
        
        if len(mask_paths) != len(data_paths):
            log.warning(f"Number of mask paths ({len(mask_paths)}) does not match number of data paths ({len(data_paths)}). Masks will not be used.")
            mask_paths = []

    # Evaluate on each data file
    results = {}
    for idx, data_path in enumerate(data_paths):
        mask_path = mask_paths[idx] if idx < len(mask_paths) else None
        log.info(f"Evaluating on {data_path}" + (f" with mask {mask_path}" if mask_path else ""))
        loss = evaluate_file(model, data_path, args.chunk_size, device, mask_path)
        
        # Extract filename without path and extension
        filename = Path(data_path).stem
        results[filename] = loss
        log.info(f"{filename}: loss = {loss:.4f}")
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()

