"""
Standalone downstream evaluation script for OLMo models.

Loads a checkpoint, builds downstream evaluators for specified tasks, runs
evaluation, and writes per-task results to a JSON file.

Usage:
    python new_utils/evaluate_downstream.py \
        --model_path /path/to/checkpoint \
        --tasks piqa,hellaswag,winogrande \
        --output_path results.json \
        [--device cuda] \
        [--batch_size 8]

"""

import argparse
import json
import logging
import math
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from olmo.config import EvaluatorConfig, EvaluatorType, TrainConfig
from olmo.eval import build_downstream_evaluator
from olmo.eval.downstream import ICLMetric, label_to_task_map
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer
from olmo.torch_util import get_global_rank, get_world_size
from hf_olmo.modeling_olmo import OLMoForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def build_evaluator_for_task(
    task_name: str,
    tokenizer: Tokenizer,
    device: torch.device,
    batch_size: int = 8,
    subset_num_batches: int = 0,
    seed: int = 42,
):
    """Build a downstream Evaluator for a single task label.

    Mirrors the logic in ``olmo.eval.build_downstream_evaluator`` so that the
    standalone script produces results consistent with in-training evaluation.
    A :class:`DistributedSampler` is created to keep behaviour identical whether
    the script is launched via ``torchrun`` (distributed) or plain ``python``
    (single-process, world_size=1, rank=0).
    """
    from olmo.eval.evaluator import Evaluator

    if task_name not in label_to_task_map:
        raise ValueError(
            f"Unknown downstream task: '{task_name}'. "
            f"Available tasks: {sorted(label_to_task_map.keys())}"
        )

    task_kwargs = {}
    task_class = label_to_task_map[task_name]
    if isinstance(task_class, tuple):
        task_class, task_kwargs = task_class

    ds_eval_dataset = task_class(tokenizer=tokenizer, **task_kwargs)

    # DistributedSampler matching olmo/eval/__init__.py and the default
    # DataConfig used by downstream evaluators in olmo_configuration.py
    # (drop_last=False, shuffle=False).
    ds_eval_sampler = DistributedSampler(
        ds_eval_dataset,
        drop_last=False,
        shuffle=False,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )

    ds_eval_dataloader = DataLoader(
        ds_eval_dataset,
        batch_size=batch_size,
        collate_fn=ds_eval_dataset.collate_fn,
        num_workers=0,
        sampler=ds_eval_sampler,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False,
        timeout=0,
    )

    metric = ICLMetric(metric_type=ds_eval_dataset.metric_type)

    evaluator = Evaluator(
        label=task_name,
        type=EvaluatorType.downstream,
        eval_loader=ds_eval_dataloader,
        eval_metric=metric.to(device),
        subset_num_batches=subset_num_batches if subset_num_batches > 0 else None,
    )
    return evaluator


def run_eval(model, evaluator, device: torch.device) -> dict:
    """Run evaluation for a single Evaluator instance and return metrics."""
    from olmo.config import EvaluatorType

    evaluator.reset_metrics()

    eval_batches = iter(evaluator.eval_loader)
    num_eval_batches = len(evaluator.eval_loader)
    if evaluator.subset_num_batches is not None and evaluator.subset_num_batches > 0:
        num_eval_batches = min(evaluator.subset_num_batches, num_eval_batches)
        eval_batches = islice(eval_batches, num_eval_batches)

    for step, batch in enumerate(eval_batches):
        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            output = model(input_ids=batch["input_ids"])
            logits = output.logits

        evaluator.update_metrics(batch, ce_loss=torch.tensor(0.0), logits=logits)

        if (step + 1) % 50 == 0 or (step + 1) == num_eval_batches:
            log.info(f"  [{evaluator.label}] eval_step={step + 1}/{num_eval_batches}")

    metrics = evaluator.compute_metrics()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate OLMo model on downstream tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to OLMo checkpoint directory")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated list of downstream task names")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--subset_num_batches", type=int, default=0, help="Limit number of eval batches per task (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--hf_model", action="store_true", help="If set, load as a HuggingFace model via OLMoForCausalLM")
    parser.add_argument("--quantize", type=int, default=None, help="Specify bits for quantized inference (4 or 8). Requires --hf_model")
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info(f"Using device: {device}")

    # Load model
    log.info(f"Loading model from {args.model_path}")
    if args.hf_model:
        quantization_config = None
        if args.quantize is not None:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=args.quantize == 8,
                load_in_4bit=args.quantize == 4,
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map="auto", dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
    else:
        model = OLMo.from_checkpoint(args.model_path, device=args.device)
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    log.info("Model loaded successfully")

    # Load tokenizer from model path
    if args.hf_model:
        log.info(f"Loading tokenizer from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        log.info(f"Loading tokenizer from Train config at {args.model_path}")
        config_path = Path(args.model_path) / "config.yaml"
        train_config = TrainConfig.load(str(config_path), validate_paths=False)
        tokenizer = Tokenizer.from_train_config(train_config)

    # Parse tasks
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    log.info(f"Tasks to evaluate: {task_names}")

    # Run evaluation
    all_results = {}
    for task_name in task_names:
        log.info(f"Evaluating task: {task_name}")
        try:
            evaluator = build_evaluator_for_task(
                task_name,
                tokenizer=tokenizer,
                device=device,
                batch_size=args.batch_size,
                subset_num_batches=args.subset_num_batches,
                seed=args.seed,
            )
            metrics = run_eval(model, evaluator, device)
            all_results.update(metrics)
            for k, v in metrics.items():
                log.info(f"  {k} = {v:.4f}")
        except Exception as e:
            log.error(f"Failed to evaluate task '{task_name}': {e}", exc_info=True)
            all_results[f"eval/downstream/{task_name}_error"] = str(e)

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
