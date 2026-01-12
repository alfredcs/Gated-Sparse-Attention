"""
Configuration classes for Gated Sparse Attention (GSA).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class GSAConfig:
    """
    Configuration for Gated Sparse Attention.

    This config controls all aspects of the GSA mechanism including:
    - Model dimensions and heads
    - Indexer configuration for sparse token selection
    - Gating mechanisms (G1 output gate, G2 value gate)
    - Adaptive sparsity settings
    - Position encoding settings
    - Optimization flags
    """

    # Model dimensions
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    d_head: Optional[int] = None  # Auto-computed if None

    # Full model config (for GSAForCausalLM)
    n_layers: int = 32
    d_ffn: int = 11008
    vocab_size: int = 128256

    # Indexer configuration
    d_indexer: int = 64
    n_indexer_heads: int = 4
    indexer_activation: str = "sigmoid"  # sigmoid, relu

    # Sparsity configuration
    k_base: int = 2048
    k_min: int = 256
    k_max: int = 4096
    use_adaptive_k: bool = True
    adaptive_k_temperature: float = 1.0

    # Gating configuration
    use_value_gate: bool = True   # G2 position (after V projection)
    use_output_gate: bool = True  # G1 position (after SDPA, most effective)
    gate_activation: str = "sigmoid"
    gate_bias_init: float = 0.5   # Initialize for moderate gating

    # Position encoding
    rope_base: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: int = 131072

    # Training stability
    dense_residual_alpha: float = 0.0  # For warm-up: blend with dense attention
    use_indexer_warmup: bool = True
    indexer_warmup_steps: int = 1000

    # Optimization
    use_flash_attention: bool = True
    use_triton_kernels: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Layer configuration
    norm_eps: float = 1e-5
    use_rms_norm: bool = True
    tie_word_embeddings: bool = False

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        """Validate and compute derived attributes."""
        if self.d_head is None:
            self.d_head = self.d_model // self.n_heads

        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.k_min <= self.k_base <= self.k_max, (
            f"k_min ({self.k_min}) <= k_base ({self.k_base}) <= k_max ({self.k_max}) required"
        )
        assert self.indexer_activation in ["sigmoid", "relu"], (
            f"indexer_activation must be 'sigmoid' or 'relu', got {self.indexer_activation}"
        )
        assert self.gate_activation in ["sigmoid", "tanh", "silu"], (
            f"gate_activation must be 'sigmoid', 'tanh', or 'silu', got {self.gate_activation}"
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GSAConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "GSAConfig":
        """Load config from pretrained model path or HuggingFace hub."""
        import json
        from pathlib import Path

        config_path = Path(model_name_or_path) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        else:
            # Try loading from HuggingFace hub
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(model_name_or_path, "config.json")
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)

    def save_pretrained(self, save_directory: str):
        """Save config to directory."""
        import json
        from pathlib import Path

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class GSATrainingConfig:
    """Configuration for GSA model training."""

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8

    # Schedule
    warmup_steps: int = 2000
    lr_scheduler: str = "cosine"
    max_steps: int = 100000

    # Batch size
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    global_batch_size: int = 256  # micro * grad_accum * num_gpus

    # Sequence length
    max_seq_len: int = 4096

    # Precision
    precision: str = "bf16"
    gradient_checkpointing: bool = True

    # Stability
    grad_clip: float = 1.0

    # GSA specific
    indexer_warmup_steps: int = 1000
    indexer_lr_multiplier: float = 10.0

    # Checkpointing
    save_interval: int = 5000
    eval_interval: int = 1000
    log_interval: int = 10
