#!/usr/bin/env python3
"""
Script to compute perplexity scores for given dataset
Uses lm-evaluation-harness for efficient perplexity computation.
Make sure to update perplexity calculation function depending on data samples used
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import os
import sys
import math
from datasets import load_dataset

# Import HFLM from lm-evaluation-harness
try:
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import loglikelihood, loglikelihood_rolling
except ImportError:
    raise ImportError(
        "Could not import lm_eval.models.huggingface. Please install lm-evaluation-harness: pip install lm-eval"
    )

import torch  # For device check


def load_hf_dataset(name: str):
    import itertools
    dataset = load_dataset(name, split="train", streaming=True).shuffle(seed=42)
    subset = list(itertools.islice(dataset, 10000))  #DEFINE SUBSET SIZE

    return subset


def format_starcoder(example):
    prompt = f"Instruction: {example['instruction']}\nResponse: "
    return {
        "prompt": prompt,
        "completion": example["text"]
    }

#USE FOR CALCULATING PPLX OVER ENTIRE STRING
def compute_perplexity_text(model, dataset: List[Dict]) -> List[Tuple[int, float]]:
    text = [sample.get('text', '') for sample in dataset]
    text_token_lengths = [len(model.tokenizer.encode(i)) for i in text]
    requests = [
        Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=(text,),
            idx=i,
        )
        for i, text in enumerate(texts)
    ]

    log_likelihoods = loglikelihood_rolling(requests)
    perplexities = [math.exp(-likelihood/length) for likelihood, length in zip(log_likelihoods, text_token_lengths)]
    
    return perplexities

#USE FOR CALCULATING PPLX ONLY ON CONTINUATION
def compute_perplexity_prompt_continutation(model, dataset: List[Dict]) -> List[Tuple[int, float]]:
    samples = [format_starcoder(sample) for sample in dataset]
    continutaton_token_lengths = [min(model.max_length, len(model.tokenizer.encode(i.get("text",)))) for i in dataset]
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(sample.get("prompt", ), sample.get("continuation", )),
            idx=i,
        )
        for i, sample in enumerate(samples)
    ]

    log_likelihoods = loglikelihood(requests)
    perplexities = [math.exp(-likelihood/length) for likelihood, length in zip(log_likelihoods, continutaton_token_lengths)]
    
    return perplexities


def main():
    parser = argparse.ArgumentParser(description="Split Alpaca dataset by perplexity quartiles (fast, using lm-eval-harness)")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/home/catheri4/experiments/alpaca_partition/alpaca_split_perplexity/datasets/alpaca_data_cleaned.json",
        help="Path to the dataset JSON file"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="allenai/OLMo-2-0425-1B",
        help="HuggingFace model name for OLMo2 1B"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model inference (increase for faster GPU inference)"
    )
    args = parser.parse_args()


    dataset = load_hf_dataset(args.dataset_path)
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        print(f"Limited to {len(dataset)} samples for testing")

    # Load HFLM model (from lm-eval-harness)
    print(f"Loading model with lm-eval-harness HFLM: {args.model_name}")
    model = HFLM(
        pretrained=args.model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.batch_size,
        use_fast_tokenizer=True,
        trust_remote_code=True,
        max_length=1024
        
    )

    perplexities = compute_perplexity_text(model, dataset)
    avg_perplexity = sum(perplexities)/len(perplexities)

    print("\n" + "="*50)
    print(f"PERPLEXITY")
    print("="*50)

    print(avg_perplexity)






if __name__ == "__main__":
    main() 