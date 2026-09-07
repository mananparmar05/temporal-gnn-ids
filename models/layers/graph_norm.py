"""
Graph Normalization Layers for GAT Encoder
Normalizes node representations within and across graph snapshots
to stabilize training and improve gradient flow in deep GAT stacks.

Implements:
  1. GraphNorm - Graph-level normalization (Cai et al., 2021)
  2. NodeNorm - Per-node feature normalization
"""

import torch
import torch.nn as nn


class GraphNorm(nn.Module):
    """
    Graph Normalization layer.
    Normalizes node features across all nodes within a single graph snapshot,
    with a learnable shift parameter to preserve graph-level statistics.
    
    GraphNorm(h) = gamma * (h - alpha * mean(h)) / std(h) + beta
    
    where alpha is a learnable parameter controlling how much of the 
    graph-level mean is subtracted (preserving graph structure information).
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Args:
            num_features: Number of node feature dimensions.
            eps: Small constant for numerical stability in division.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Learnable affine parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Learnable mean-shift coefficient (key innovation of GraphNorm)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix [num_nodes, num_features]
            batch: Batch assignment vector [num_nodes] indicating which graph 
                   each node belongs to. If None, treats all nodes as one graph.
        
        Returns:
            Normalized node features [num_nodes, num_features]
        """
        if batch is None:
            # Single graph: normalize across all nodes
            mean = x.mean(dim=0, keepdim=True)
            x = x - self.alpha * mean
            var = x.var(dim=0, keepdim=True, unbiased=False)
            x_norm = x / torch.sqrt(var + self.eps)
        else:
            # Batched graphs: normalize per-graph
            unique_graphs = batch.unique()
            x_norm = torch.zeros_like(x)
            
            for g_id in unique_graphs:
                mask = (batch == g_id)
                x_g = x[mask]
                mean = x_g.mean(dim=0, keepdim=True)
                x_g = x_g - self.alpha * mean
                var = x_g.var(dim=0, keepdim=True, unbiased=False)
                x_norm[mask] = x_g / torch.sqrt(var + self.eps)
        
        return self.gamma * x_norm + self.beta


class NodeNorm(nn.Module):
    """
    Per-Node Feature Normalization.
    Normalizes each node's feature vector independently across the feature dimension.
    Simpler alternative to GraphNorm for shallow architectures.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_features, eps=eps)
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix [num_nodes, num_features] or [batch, nodes, features]
        
        Returns:
            Normalized features with same shape as input.
        """
        return self.layer_norm(x)
