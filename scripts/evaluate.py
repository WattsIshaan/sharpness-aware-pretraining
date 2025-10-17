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


def evaluate_file(model: OLMo, data_path: str, chunk_size: int, device: torch.device) -> float:
    """Evaluate model on a single data file."""
    # Load data using memmap
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    
    total_loss = 0.0
    total_tokens = 0
    
    # Process data in chunks
    num_chunks = len(data) // chunk_size
    
    log.info(f"Processing {num_chunks} chunks from {data_path}")
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = data[start_idx:end_idx]
        
        # Convert to tensor and add batch dimension
        input_ids = torch.from_numpy(np.array(chunk)).long().unsqueeze(0).to(device)
        
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

    # Evaluate on each data file
    results = {}
    for data_path in data_paths:
        log.info(f"Evaluating on {data_path}")
        loss = evaluate_file(model, data_path, args.chunk_size, device)
        
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

