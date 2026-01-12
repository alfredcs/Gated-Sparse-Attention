"""
Seed and distributed utilities.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rank() -> int:
    """Get current process rank in distributed setting."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Get total number of processes in distributed setting."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    """Get local rank within current node."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process."""
    return get_rank() == 0
