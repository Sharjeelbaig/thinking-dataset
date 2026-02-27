"""NeuroThinker configuration."""
from dataclasses import dataclass

@dataclass
class NeuroThinkerConfig:
    vocab_size: int = 32000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_head: int = 64
    d_ff: int = 720
    d_memory: int = 192
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0
    memory_decay_init: float = 0.99
    surprise_threshold: float = 0.1
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    model_type: str = "neurothinker"

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
