"""
Positional Encoding Layers for Temporal Snapshot Sequences
Injects temporal ordering information into snapshot embeddings so the 
Temporal Self-Attention Encoder can distinguish snapshot chronology.

Implements two variants:
  1. Sinusoidal (fixed, no learnable parameters) - Vaswani et al., 2017
  2. Learned (trainable embedding table)
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from 'Attention Is All You Need'.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    No learnable parameters; the encoding is computed once and cached.
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension (must match temporal encoder input).
            max_len: Maximum sequence length supported.
            dropout: Dropout applied after adding positional encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] for broadcasting
        
        # Register as buffer (not a parameter, but moves with .to(device))
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Positionally encoded tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding using a trainable embedding table.
    Each position index maps to a learnable d_model-dimensional vector.
    More flexible than sinusoidal but requires more parameters.
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout after encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Initialize with Xavier uniform for stable gradients
        nn.init.xavier_uniform_(self.position_embedding.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Positionally encoded tensor [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        return self.dropout(x)
