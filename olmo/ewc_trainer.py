"""Elastic Weight Consolidation (EWC) training on top of the OLMo ``Trainer``.

EWC adds a quadratic penalty that anchors parameters to a reference checkpoint
``θ*`` using a diagonal Fisher estimate ``F`` (Kirkpatrick et al., PNAS 2017):

    L_total = L_task + (λ/2) Σ_i F_i (θ_i - θ*_i)²

``θ*`` is taken from model weights immediately after loading the pretrained
checkpoint. ``F`` is estimated as the running mean of squared gradients of the
CE loss on ``ewc_fisher_batches`` batches from a **separate** dataloader (so the
main SFT iterator is not advanced). If ``ewc_fisher_paths`` is set (e.g. subsampled pretrain
**train** memmaps), those are used; otherwise the SFT training paths are used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from olmo.config import DistributedStrategy
from olmo.data import build_train_dataloader
from olmo.torch_util import barrier, get_world_size, move_to_device
from olmo.train import Trainer

log = logging.getLogger(__name__)


@dataclass
class EWCTrainer(Trainer):
    """Trainer that adds an EWC penalty to each micro-batch after Fisher is estimated."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._ewc_theta_star: Optional[Dict[torch.nn.Parameter, torch.Tensor]] = None
        self._ewc_fisher: Optional[Dict[torch.nn.Parameter, torch.Tensor]] = None
        self._ewc_lambda_active: Optional[float] = None

    def prepare_ewc(self) -> None:
        """Snapshot ``θ*``, estimate diagonal Fisher, then enable the EWC term."""
        cfg = self.cfg
        if cfg.ewc_lambda is None:
            raise ValueError("prepare_ewc requires cfg.ewc_lambda to be set")

        if cfg.distributed_strategy not in (DistributedStrategy.ddp, DistributedStrategy.single):
            raise NotImplementedError(
                "EWC is only implemented for DDP or single-device training; FSDP support is not implemented."
            )

        self._ewc_lambda_active = None

        self._ewc_theta_star = {}
        for p in self.model.parameters():
            if p.requires_grad:
                self._ewc_theta_star[p] = p.detach().clone()

        fisher: Dict[torch.nn.Parameter, torch.Tensor] = {
            p: torch.zeros_like(p) for p in self._ewc_theta_star
        }

        if cfg.ewc_fisher_paths:
            # Fisher data often uses different memmaps than SFT training. Both dataloaders default
            # to ``save_folder / "train_data" / "global_indices.npy"``; reusing the same folder
            # would overwrite SFT indices and break the main train loader. Use an auxiliary folder.
            fisher_save_folder = str(Path(cfg.save_folder) / "_ewc_fisher_indices")
            fisher_cfg = replace(
                cfg,
                save_folder=fisher_save_folder,
                data=replace(
                    cfg.data,
                    paths=list(cfg.ewc_fisher_paths),
                    label_mask_paths=cfg.ewc_fisher_label_mask_paths,
                ),
            )
            log.info(
                "EWC: estimating Fisher diagonal over %d batches (λ=%s) on %d memmap path(s) "
                "(ewc_fisher_paths, e.g. pretrain val); Fisher indices under %s/train_data ...",
                cfg.ewc_fisher_batches,
                cfg.ewc_lambda,
                len(cfg.ewc_fisher_paths),
                fisher_save_folder,
            )
            fisher_loader = build_train_dataloader(fisher_cfg)
        else:
            log.info(
                "EWC: estimating Fisher diagonal over %d batches (λ=%s) on SFT training data...",
                cfg.ewc_fisher_batches,
                cfg.ewc_lambda,
            )
            fisher_loader = build_train_dataloader(cfg)
        it = iter(fisher_loader)
        self.dist_model.train()

        for i in range(cfg.ewc_fisher_batches):
            self.optim.zero_grad(set_to_none=True)
            batch = next(it)
            batch = move_to_device(batch, self.device)
            ce_batch_loss, _ = self.train_batch(batch)
            del ce_batch_loss
            if self._grads_are_finite():
                for p in fisher:
                    if p.grad is not None:
                        fisher[p] += p.grad.detach() ** 2
            else:
                log.warning("EWC Fisher batch %d: skipping due to non-finite gradients", i)
            self.optim.zero_grad(set_to_none=True)

        del fisher_loader

        for p in fisher:
            fisher[p] /= float(cfg.ewc_fisher_batches)

        if dist.is_initialized() and get_world_size() > 1:
            for p in fisher:
                dist.all_reduce(fisher[p], op=dist.ReduceOp.SUM)
                fisher[p] /= float(get_world_size())

        self._ewc_fisher = fisher
        self._ewc_lambda_active = cfg.ewc_lambda
        barrier()
        log.info("EWC: Fisher estimation complete; training with consolidation penalty.")

    def _grads_are_finite(self) -> bool:
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
        return True

    def train_micro_batch(
        self, micro_batch: Dict[str, Any], batch_size_in_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        loss, ce_loss, z_loss = super().train_micro_batch(micro_batch, batch_size_in_tokens)
        if (
            self._ewc_lambda_active is not None
            and self._ewc_fisher is not None
            and self._ewc_theta_star is not None
        ):
            ewc_term = torch.zeros((), device=self.device, dtype=torch.float32)
            for p in self._ewc_theta_star:
                ewc_term = ewc_term + (self._ewc_fisher[p] * (p - self._ewc_theta_star[p]).pow(2)).sum()
            loss = loss + (self._ewc_lambda_active / 2.0) * ewc_term.to(dtype=loss.dtype)
        return loss, ce_loss, z_loss
