"""Multi-Head Attention with Rotary Position Embeddings (RoPE)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0,
                          device: torch.device = None) -> torch.Tensor:
    """Precompute complex exponential frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, d_head//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.
    
    Args:
        x: (batch, n_heads, seq_len, d_head)
        freqs: (seq_len, d_head//2) complex
    """
    # Reshape x to pairs: (batch, n_heads, seq_len, d_head//2, 2)
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_pairs)
    
    # Broadcast freqs: (1, 1, seq_len, d_head//2)
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs[:, :, :x_complex.shape[2], :]
    
    # Back to real
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)


class RotaryMultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE (no learned position embeddings)."""

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 max_seq_len: int = 512, dropout: float = 0.1,
                 rope_theta: float = 10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5

        self.w_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.w_o = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(d_head, max_seq_len, rope_theta),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.w_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to queries and keys
        q = apply_rope(q, self.rope_freqs[:T].to(x.device))
        k = apply_rope(k, self.rope_freqs[:T].to(x.device))

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.w_o(out))
