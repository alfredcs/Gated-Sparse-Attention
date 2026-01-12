"""
Optimizer and scheduler creation utilities.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Any, Dict, Optional


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Create optimizer with weight decay.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        Configured optimizer
    """
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "layernorm" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Separate indexer parameters for higher LR
    indexer_decay_params = []
    indexer_no_decay_params = []
    other_decay_params = []
    other_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_indexer = "indexer" in name
        is_no_decay = "bias" in name or "layernorm" in name or "norm" in name

        if is_indexer:
            if is_no_decay:
                indexer_no_decay_params.append(param)
            else:
                indexer_decay_params.append(param)
        else:
            if is_no_decay:
                other_no_decay_params.append(param)
            else:
                other_decay_params.append(param)

    # Get LR multiplier for indexer
    indexer_lr_multiplier = config.get("indexer_lr_multiplier", 1.0)
    base_lr = config.get("learning_rate", 3e-4)
    weight_decay = config.get("weight_decay", 0.1)

    param_groups = [
        {
            "params": other_decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr,
        },
        {
            "params": other_no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr,
        },
        {
            "params": indexer_decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr * indexer_lr_multiplier,
        },
        {
            "params": indexer_no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr * indexer_lr_multiplier,
        },
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = AdamW(
        param_groups,
        lr=base_lr,
        betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
        eps=config.get("eps", 1e-8),
    )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float = 1e-6,
    schedule_type: str = "cosine",
) -> LambdaLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr: Minimum learning rate
        schedule_type: Type of schedule ("cosine", "linear", "constant")

    Returns:
        LR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        # Get base LR from optimizer (first group)
        base_lr = optimizer.defaults["lr"]

        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Post-warmup phase
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))

        if schedule_type == "cosine":
            # Cosine decay
            return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif schedule_type == "linear":
            # Linear decay
            return max(min_lr / base_lr, 1.0 - progress)
        elif schedule_type == "constant":
            return 1.0
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    return LambdaLR(optimizer, lr_lambda)


class WarmupCosineDecayScheduler(LambdaLR):
    """
    Learning rate scheduler with warmup and cosine decay.

    More configurable version with explicit min_lr_ratio parameter.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))

        progress = float(current_step - self.warmup_steps) / float(
            max(1, self.max_steps - self.warmup_steps)
        )
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
