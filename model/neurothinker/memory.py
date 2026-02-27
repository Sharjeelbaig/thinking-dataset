"""Titans-inspired Memory Module.

Implements the key ideas from Google's Titans architecture:
1. Deep MLP memory (not a flat vector/matrix like traditional RNNs)
2. Surprise gate — detects novel inputs via gradient-based signals
3. Forgetting gate — learnable adaptive decay for memory capacity management
4. Momentum — smooths surprise signals to capture trending context
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TitansMemoryModule(nn.Module):
    """Long-term memory module inspired by Google Titans architecture.
    
    The memory is a small MLP whose parameters represent compressed
    knowledge. It updates itself based on how "surprising" new inputs
    are (high gradient norm = novel information → store it).
    """

    def __init__(self, d_model: int, d_memory: int,
                 decay_init: float = 0.99, dropout: float = 0.1):
        super().__init__()

        # Deep memory MLP (the "long-term memory storage")
        self.memory_net = nn.Sequential(
            nn.Linear(d_model, d_memory, bias=False),
            nn.SiLU(),
            nn.Linear(d_memory, d_model, bias=False),
        )

        # Surprise gate — learns to detect when input is novel
        self.surprise_gate = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Sigmoid(),
        )

        # Forgetting gate — per-dimension learnable decay
        # Initialized near 1.0 so memory starts by retaining most info
        self.forget_bias = nn.Parameter(torch.full((d_model,), decay_init))

        # Momentum for smoothing surprise signals (EMA)
        self.momentum = nn.Parameter(torch.tensor(0.9))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Running surprise EMA (not a learned parameter)
        self.register_buffer("surprise_ema", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the memory module.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            Memory-enhanced representation (batch, seq_len, d_model)
        """
        B, T, D = x.shape

        # Query the memory with current input
        memory_out = self.memory_net(x)  # What memory "knows" about this input

        # Compute surprise: how different is the input from what memory expects?
        surprise_signal = torch.norm(x - memory_out, dim=-1, keepdim=True)
        surprise_signal = surprise_signal / (surprise_signal.mean() + 1e-8)

        # Apply momentum to smooth the surprise signal
        momentum = torch.sigmoid(self.momentum)
        smoothed_surprise = momentum * self.surprise_ema + (1 - momentum) * surprise_signal.mean()
        self.surprise_ema = smoothed_surprise.detach()

        # Surprise gate — modulates how much new info to incorporate
        gate = self.surprise_gate(x)  # (B, T, D)

        # Scale gate by normalized surprise
        gate = gate * torch.clamp(surprise_signal, 0, 2)

        # Forgetting gate — how much of old memory to retain
        forget = torch.sigmoid(self.forget_bias).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

        # Combine: forgotten old memory + gated new information
        updated = forget * memory_out + gate * x

        # Project and normalize
        out = self.out_proj(updated)
        out = self.dropout(out)

        return self.norm(out + x)  # Residual connection
