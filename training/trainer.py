"""
GSA Trainer class for managing training loop.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import wandb

from gsa.utils import get_rank, save_checkpoint, load_checkpoint


class GSATrainer:
    """
    Trainer class for GSA models.

    Handles:
    - Training loop with gradient accumulation
    - Mixed precision training
    - Gradient clipping
    - Checkpointing
    - Logging
    - Indexer warmup
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: Dict[str, Any],
        output_dir: Path,
        is_distributed: bool = False,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            scheduler: LR scheduler
            config: Training configuration
            output_dir: Directory for checkpoints
            is_distributed: Whether using distributed training
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output_dir = Path(output_dir)
        self.is_distributed = is_distributed
        self.rank = get_rank()

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Mixed precision
        self.use_amp = config.get("precision", "bf16") in ["bf16", "fp16"]
        self.amp_dtype = torch.bfloat16 if config.get("precision") == "bf16" else torch.float16
        self.scaler = GradScaler() if config.get("precision") == "fp16" else None

        # Gradient accumulation
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # Gradient clipping
        self.grad_clip = config.get("grad_clip", 1.0)

        # Indexer warmup
        self.indexer_warmup_steps = config.get("indexer_warmup_steps", 1000)
        self.indexer_lr_multiplier = config.get("indexer_lr_multiplier", 10.0)

        # Logging
        self.log_interval = config.get("log_interval", 10)
        self.eval_interval = config.get("eval_interval", 1000)
        self.save_interval = config.get("save_interval", 5000)

        # Gradient checkpointing
        if config.get("gradient_checkpointing", False):
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        if hasattr(model, "model"):
            model.model.gradient_checkpointing = True

    def _get_indexer_params(self):
        """Get indexer parameters for special LR treatment."""
        indexer_params = []
        other_params = []

        model = self.model.module if hasattr(self.model, "module") else self.model

        for name, param in model.named_parameters():
            if "indexer" in name:
                indexer_params.append(param)
            else:
                other_params.append(param)

        return indexer_params, other_params

    def train(self, dataloader: DataLoader, max_steps: int):
        """
        Main training loop.

        Args:
            dataloader: Training data loader
            max_steps: Maximum number of training steps
        """
        self.model.train()

        # Progress bar (only on rank 0)
        if self.rank == 0:
            pbar = tqdm(total=max_steps, initial=self.global_step, desc="Training")
        else:
            pbar = None

        # Metrics accumulation
        accumulated_loss = 0.0
        accumulated_samples = 0

        # Data iterator
        data_iter = iter(dataloader)

        while self.global_step < max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move to device
            input_ids = batch["input_ids"].cuda()
            labels = batch.get("labels", input_ids.clone())
            if isinstance(labels, torch.Tensor):
                labels = labels.cuda()

            # Forward pass with mixed precision
            with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item() * self.gradient_accumulation_steps
            accumulated_samples += 1

            # Update weights after accumulation
            if accumulated_samples % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                else:
                    grad_norm = self._compute_grad_norm()

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.log_interval == 0 and self.rank == 0:
                    avg_loss = accumulated_loss / self.log_interval
                    lr = self.scheduler.get_last_lr()[0]

                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": lr,
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "step": self.global_step,
                        "epoch": self.epoch,
                    })

                    if pbar is not None:
                        pbar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        })
                        pbar.update(self.log_interval)

                    accumulated_loss = 0.0

                # Save checkpoint
                if self.global_step % self.save_interval == 0 and self.rank == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

        if pbar is not None:
            pbar.close()

    def _compute_grad_norm(self) -> float:
        """Compute gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        model = self.model.module if hasattr(self.model, "module") else self.model

        save_checkpoint(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            path=self.output_dir / name,
            config=self.config,
            extra_state={
                "epoch": self.epoch,
            },
        )

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        model = self.model.module if hasattr(self.model, "module") else self.model

        state = load_checkpoint(
            path=path,
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.global_step = state["step"]
        if state.get("extra_state"):
            self.epoch = state["extra_state"].get("epoch", 0)

        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
