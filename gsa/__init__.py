"""
Gated Sparse Attention (GSA) - A novel attention mechanism combining
sparse token selection with dual gating for efficient long-context processing.
"""

from .config import GSAConfig
from .attention.gated_sparse_attention import GatedSparseAttention
from .attention.gated_indexer import GatedLightningIndexer
from .attention.adaptive_topk import AdaptiveTopKSelector
from .attention.value_gate import ValueGate
from .attention.output_gate import OutputGate
from .attention.rope import RotaryEmbedding
from .models.gsa_llama import GSALlamaForCausalLM, GSAForCausalLM
from .utils.checkpoint import load_checkpoint, save_checkpoint
from .utils.convert import replace_attention_with_gsa

__version__ = "0.1.0"
__all__ = [
    "GSAConfig",
    "GatedSparseAttention",
    "GatedLightningIndexer",
    "AdaptiveTopKSelector",
    "ValueGate",
    "OutputGate",
    "RotaryEmbedding",
    "GSALlamaForCausalLM",
    "GSAForCausalLM",
    "load_checkpoint",
    "save_checkpoint",
    "replace_attention_with_gsa",
]
