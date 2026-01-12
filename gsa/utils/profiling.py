"""
Profiling utilities for GSA performance analysis.
"""

import torch
import time
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager


@contextmanager
def cuda_sync_timer():
    """Context manager for accurate GPU timing."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


def profile_memory(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """
    Profile memory usage of a model.

    Args:
        model: Model to profile
        input_shape: Input tensor shape (batch, seq_len)
        device: Device to run on
        dtype: Data type

    Returns:
        Dictionary with memory statistics
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    batch_size, seq_len = input_shape

    # Create input
    input_ids = torch.randint(
        0, model.config.vocab_size if hasattr(model, 'config') else 32000,
        input_shape,
        device=device,
    )

    # Forward pass
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        _ = model(input_ids)

    torch.cuda.synchronize()

    # Get memory stats
    allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.max_memory_reserved() / 1024**3

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "peak_allocated_gb": allocated,
        "peak_reserved_gb": reserved,
        "dtype": str(dtype),
    }


def profile_speed(
    model: torch.nn.Module,
    input_shape: tuple,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    include_backward: bool = False,
) -> Dict[str, Any]:
    """
    Profile inference/training speed of a model.

    Args:
        model: Model to profile
        input_shape: Input tensor shape (batch, seq_len)
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        device: Device to run on
        dtype: Data type
        include_backward: Whether to include backward pass

    Returns:
        Dictionary with timing statistics
    """
    batch_size, seq_len = input_shape

    # Create input
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32000
    input_ids = torch.randint(0, vocab_size, input_shape, device=device)

    if include_backward:
        labels = torch.randint(0, vocab_size, input_shape, device=device)
        model.train()
    else:
        model.eval()

    # Warmup
    for _ in range(num_warmup):
        with torch.autocast(device_type="cuda", dtype=dtype):
            if include_backward:
                outputs = model(input_ids, labels=labels)
                outputs.loss.backward()
            else:
                with torch.no_grad():
                    _ = model(input_ids)
        torch.cuda.synchronize()

    # Timed runs
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        with torch.autocast(device_type="cuda", dtype=dtype):
            if include_backward:
                outputs = model(input_ids, labels=labels)
                outputs.loss.backward()
            else:
                with torch.no_grad():
                    _ = model(input_ids)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    # Compute statistics
    avg_time = total_time / num_iterations
    tokens_per_second = (batch_size * seq_len) / avg_time

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "avg_time_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_second,
        "include_backward": include_backward,
        "dtype": str(dtype),
        "num_iterations": num_iterations,
    }


def compare_attention_methods(
    seq_lengths: list = [1024, 2048, 4096, 8192, 16384],
    batch_size: int = 2,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    k_selected: int = 2048,
    device: str = "cuda",
) -> Dict[str, Dict[int, float]]:
    """
    Compare performance of different attention methods.

    Args:
        seq_lengths: List of sequence lengths to test
        batch_size: Batch size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of KV heads
        k_selected: Number of selected tokens for sparse attention
        device: Device to run on

    Returns:
        Dictionary mapping method names to {seq_len: time_ms}
    """
    from ..config import GSAConfig
    from ..attention import GatedSparseAttention

    results = {
        "standard_sdpa": {},
        "gsa": {},
    }

    for seq_len in seq_lengths:
        print(f"\nTesting seq_len={seq_len}...")

        # Create inputs
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Standard SDPA
        q = torch.randn(batch_size, seq_len, n_heads, d_model // n_heads, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch_size, seq_len, n_kv_heads, d_model // n_heads, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, seq_len, n_kv_heads, d_model // n_heads, device=device, dtype=torch.bfloat16)

        # Warmup and time standard SDPA
        torch.cuda.synchronize()
        for _ in range(3):
            with torch.no_grad():
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
                )
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
                )
        torch.cuda.synchronize()
        results["standard_sdpa"][seq_len] = (time.perf_counter() - start) / 10 * 1000

        # GSA
        config = GSAConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            k_base=min(k_selected, seq_len),
        )
        gsa = GatedSparseAttention(config).to(device).to(torch.bfloat16)

        # Warmup and time GSA
        torch.cuda.synchronize()
        for _ in range(3):
            with torch.no_grad():
                _ = gsa(x, positions=positions)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                _ = gsa(x, positions=positions)
        torch.cuda.synchronize()
        results["gsa"][seq_len] = (time.perf_counter() - start) / 10 * 1000

        print(f"  Standard SDPA: {results['standard_sdpa'][seq_len]:.2f} ms")
        print(f"  GSA: {results['gsa'][seq_len]:.2f} ms")
        print(f"  Speedup: {results['standard_sdpa'][seq_len] / results['gsa'][seq_len]:.2f}x")

    return results
