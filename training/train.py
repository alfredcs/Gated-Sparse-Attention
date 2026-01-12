#!/usr/bin/env python3
"""
Main training script for GSA models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from gsa import GSAForCausalLM, GSAConfig
from gsa.utils import set_seed, get_rank, get_world_size
from training.trainer import GSATrainer
from training.data import create_dataloader
from training.optimizer import create_optimizer, create_scheduler


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GSA model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def main():
    args = parse_args()

    # Setup distributed
    is_distributed = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    # Load config
    config = OmegaConf.load(args.config)

    # Set seed
    set_seed(args.seed + rank)

    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if rank == 0:
        wandb.init(
            project=config.logging.get("project", "gsa-pretrain"),
            name=args.wandb_run_name,
            config=OmegaConf.to_container(config),
        )

    # Create model config
    logger.info(f"Creating GSA model...")
    model_config = GSAConfig(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        n_kv_heads=config.model.n_kv_heads,
        d_ffn=config.model.d_ffn,
        vocab_size=config.model.vocab_size,
        max_position_embeddings=config.model.max_position_embeddings,
        # GSA specific
        d_indexer=config.model.gsa.d_indexer,
        n_indexer_heads=config.model.gsa.n_indexer_heads,
        k_base=config.model.gsa.k_base,
        k_min=config.model.gsa.k_min,
        k_max=config.model.gsa.k_max,
        use_value_gate=config.model.gsa.use_value_gate,
        use_output_gate=config.model.gsa.use_output_gate,
        use_adaptive_k=config.model.gsa.use_adaptive_k,
    )

    model = GSAForCausalLM(model_config)
    model = model.cuda()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    # Wrap for distributed
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,  # GSA has conditional components
        )

    # Create dataloader
    train_dataloader = create_dataloader(
        config.data,
        batch_size=config.training.micro_batch_size,
        max_seq_len=config.training.max_seq_len,
        num_workers=config.data.num_workers,
        rank=rank,
        world_size=world_size,
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config.training)
    scheduler = create_scheduler(
        optimizer,
        config.training.warmup_steps,
        config.training.max_steps,
        config.training.get("min_learning_rate", 1e-6),
    )

    # Create trainer
    trainer = GSATrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config.training,
        output_dir=output_dir,
        is_distributed=is_distributed,
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    # Training loop
    logger.info("Starting training...")
    trainer.train(train_dataloader, config.training.max_steps)

    # Save final checkpoint
    if rank == 0:
        trainer.save_checkpoint("final")
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
