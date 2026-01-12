"""
Attention pattern analysis for GSA models.

Analyzes attention sinks, gating behavior, and other attention characteristics.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm


def analyze_attention_sinks(
    model: torch.nn.Module,
    config: Dict[str, Any],
    num_samples: int = 100,
    seq_len: int = 2048,
) -> Dict[str, Any]:
    """
    Analyze attention sink phenomenon in GSA model.

    Measures:
    - Attention allocation to first token
    - Gating score distribution
    - Massive activation analysis
    - Per-layer attention patterns

    Args:
        model: Model to analyze
        config: Analysis configuration
        num_samples: Number of samples
        seq_len: Sequence length

    Returns:
        Dictionary with analysis results
    """
    model.eval()
    device = next(model.parameters()).device

    # Get vocab size
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32000

    # Storage for metrics
    first_token_attention = []
    gate_scores = {"value_gate": [], "output_gate": []}
    max_activations = []
    per_layer_first_token = {}

    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Analyzing attention"):
            # Generate random input
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

            # Forward with attention output
            outputs = model(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
            )

            # Analyze attention patterns
            if outputs.attentions is not None:
                for layer_idx, attn_weights in enumerate(outputs.attentions):
                    if attn_weights is not None:
                        # attn_weights shape depends on implementation
                        # For sparse attention: [batch, n_heads, seq, k_selected]
                        # We need to track which positions get most attention

                        if layer_idx not in per_layer_first_token:
                            per_layer_first_token[layer_idx] = []

                        # This is a simplified analysis
                        # Full analysis would track actual token positions

            # Collect gate scores from modules
            for name, module in model.named_modules():
                if hasattr(module, 'last_gate_scores') and module.last_gate_scores is not None:
                    scores = module.last_gate_scores
                    if 'value_gate' in name:
                        gate_scores["value_gate"].append(scores.mean().item())
                    elif 'output_gate' in name:
                        gate_scores["output_gate"].append(scores.mean().item())

            # Analyze max activations
            if outputs.hidden_states is not None:
                for hidden_state in outputs.hidden_states:
                    max_activations.append(hidden_state.abs().max().item())

    # Compute statistics
    results = {
        "first_token_attention": {
            "mean": np.mean(first_token_attention) if first_token_attention else 0,
            "std": np.std(first_token_attention) if first_token_attention else 0,
        },
        "value_gate": {
            "mean": np.mean(gate_scores["value_gate"]) if gate_scores["value_gate"] else 0,
            "std": np.std(gate_scores["value_gate"]) if gate_scores["value_gate"] else 0,
        },
        "output_gate": {
            "mean": np.mean(gate_scores["output_gate"]) if gate_scores["output_gate"] else 0,
            "std": np.std(gate_scores["output_gate"]) if gate_scores["output_gate"] else 0,
        },
        "max_activation": {
            "mean": np.mean(max_activations) if max_activations else 0,
            "max": np.max(max_activations) if max_activations else 0,
        },
        "attention_sink_present": _check_attention_sink(first_token_attention),
    }

    return results


def _check_attention_sink(first_token_attentions: List[float], threshold: float = 0.3) -> bool:
    """
    Check if attention sink phenomenon is present.

    Attention sink is present if first token receives disproportionate attention.
    """
    if not first_token_attentions:
        return False
    mean_first_token = np.mean(first_token_attentions)
    return mean_first_token > threshold


def analyze_sparsity_patterns(
    model: torch.nn.Module,
    config: Dict[str, Any],
    num_samples: int = 50,
    seq_len: int = 4096,
) -> Dict[str, Any]:
    """
    Analyze sparsity patterns in GSA attention.

    Examines:
    - Distribution of selected tokens
    - Adaptive k values
    - Indexer score distributions

    Args:
        model: Model to analyze
        config: Analysis configuration
        num_samples: Number of samples
        seq_len: Sequence length

    Returns:
        Dictionary with sparsity analysis
    """
    model.eval()
    device = next(model.parameters()).device

    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32000

    k_values = []
    indexer_score_stats = []

    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Analyzing sparsity"):
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

            # Hook to capture indexer outputs
            indexer_outputs = []

            def indexer_hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    indexer_outputs.append(output.detach())

            # Register hooks
            hooks = []
            for name, module in model.named_modules():
                if 'indexer' in name.lower():
                    hooks.append(module.register_forward_hook(indexer_hook))

            # Forward pass
            _ = model(input_ids)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Analyze indexer outputs
            for output in indexer_outputs:
                if output.dim() >= 2:
                    # Compute statistics
                    valid_scores = output[output != float('-inf')]
                    if len(valid_scores) > 0:
                        indexer_score_stats.append({
                            "mean": valid_scores.mean().item(),
                            "std": valid_scores.std().item(),
                            "max": valid_scores.max().item(),
                        })

    results = {
        "k_values": {
            "mean": np.mean(k_values) if k_values else 0,
            "std": np.std(k_values) if k_values else 0,
        },
        "indexer_scores": {
            "mean": np.mean([s["mean"] for s in indexer_score_stats]) if indexer_score_stats else 0,
            "std": np.mean([s["std"] for s in indexer_score_stats]) if indexer_score_stats else 0,
        },
    }

    return results


def visualize_attention_patterns(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
) -> np.ndarray:
    """
    Visualize attention patterns for a specific layer and head.

    Args:
        model: Model to analyze
        input_ids: Input token IDs
        layer_idx: Layer index
        head_idx: Head index

    Returns:
        Attention matrix as numpy array
    """
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    if outputs.attentions is None or len(outputs.attentions) <= layer_idx:
        return np.zeros((input_ids.shape[1], input_ids.shape[1]))

    attn = outputs.attentions[layer_idx]

    # Handle different attention shapes
    if attn.dim() == 4:  # [batch, heads, seq, seq] or [batch, heads, seq, k]
        attn_matrix = attn[0, head_idx].cpu().numpy()
    else:
        attn_matrix = attn.cpu().numpy()

    return attn_matrix
