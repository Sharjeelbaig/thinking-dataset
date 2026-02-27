"""NeuroThinker — Complete model architecture.

A Titans-inspired small language model with:
- RoPE positional encoding (no learned positions)
- SwiGLU feed-forward (not GELU)
- RMSNorm pre-normalization (not LayerNorm post-norm)
- Titans memory module per layer (surprise + forgetting gates)
- Causal (autoregressive) attention
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .config import NeuroThinkerConfig
from .attention import RotaryMultiHeadAttention
from .memory import TitansMemoryModule
from .layers import RMSNorm, SwiGLUFFN


class NeuroThinkerBlock(nn.Module):
    """Single transformer block: Norm→Attn→Norm→Memory→Norm→FFN."""

    def __init__(self, config: NeuroThinkerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = RotaryMultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
        )
        self.memory_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.memory = TitansMemoryModule(
            d_model=config.d_model,
            d_memory=config.d_memory,
            decay_init=config.memory_decay_init,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.attn_norm(x), mask=mask)
        # Memory module (has its own internal residual)
        x = self.memory(self.memory_norm(x))
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NeuroThinkerModel(nn.Module):
    """NeuroThinker: Titans-inspired Small Language Model."""

    def __init__(self, config: NeuroThinkerConfig):
        super().__init__()
        self.config = config

        # Token embedding (no position embedding — RoPE handles it)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks with memory
        self.blocks = nn.ModuleList([
            NeuroThinkerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm + LM head
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)
        print(f"NeuroThinker initialized: {self.num_parameters()/1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor = None) -> dict:
        """Forward pass.
        
        Args:
            input_ids: (batch, seq_len) token indices
            labels: (batch, seq_len) target token indices for loss
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        x = self.token_emb(input_ids)
        mask = self._make_causal_mask(T, x.device)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128,
                 temperature: float = 0.7, top_k: int = 50,
                 top_p: float = 0.9, eos_token_id: int = None) -> torch.Tensor:
        """Autoregressive text generation with top-k/top-p sampling."""
        self.eval()
        eos_token_id = eos_token_id or self.config.eos_token_id

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits = self(idx_cond)["logits"][:, -1, :]

            if temperature > 0:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def save_pretrained(self, save_dir: str):
        """Save model weights and config."""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        print(f"Model saved to {path}")

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu"):
        """Load model from saved weights and config."""
        path = Path(load_dir)
        with open(path / "config.json") as f:
            config = NeuroThinkerConfig.from_dict(json.load(f))
        model = cls(config)
        state = torch.load(path / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device)
