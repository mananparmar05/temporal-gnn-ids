"""
Per-Snapshot Spatial Graph Attention Network (GAT) Encoder
Section 6.2 Implementation:
Computes node embeddings h_i(t) by attending over each host's neighbors:
h_i(t) = \sigma( \sum_{j \in N(i)} \alpha_{ij}(t) \cdot W \cdot x_j(t) )
"""

import numpy as np

def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -10, 10)) - 1))

class SpatialGATLayerNumPy:
    def __init__(self, in_features, out_features, heads=2, seed=42):
        np.random.seed(seed)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        
        # Xavier uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, size=(in_features, heads * out_features)).astype(np.float32)
        self.att_src = np.random.uniform(-limit, limit, size=(1, heads, out_features)).astype(np.float32)
        self.att_dst = np.random.uniform(-limit, limit, size=(1, heads, out_features)).astype(np.float32)
        self.bias = np.zeros(heads * out_features, dtype=np.float32)

    def forward(self, x, edge_index):
        N = x.shape[0]
        E = edge_index.shape[1] if len(edge_index.shape) > 1 else 0
        
        # Linear projection h = x * W -> (N, heads, out_features)
        h = np.dot(x, self.W).reshape(N, self.heads, self.out_features)
        
        if E == 0:
            out = h.reshape(N, self.heads * self.out_features) + self.bias
            return elu(out)
            
        src, dst = edge_index[0], edge_index[1]
        
        alpha_src = np.sum(h * self.att_src, axis=-1) # (N, H)
        alpha_dst = np.sum(h * self.att_dst, axis=-1) # (N, H)
        
        edge_alpha = alpha_src[src] + alpha_dst[dst] # (E, H)
        edge_alpha = leaky_relu(edge_alpha)
        
        # Softmax over incoming neighbors
        alpha_exp = np.exp(edge_alpha - np.max(edge_alpha, axis=0, keepdims=True))
        
        alpha_sum = np.zeros((N, self.heads), dtype=np.float32)
        np.add.at(alpha_sum, dst, alpha_exp)
        alpha_norm = alpha_exp / (alpha_sum[dst] + 1e-8) # (E, H)
        
        # Message passing aggregation
        msg = h[src] * alpha_norm[:, :, np.newaxis] # (E, H, F_out)
        h_out = np.zeros((N, self.heads, self.out_features), dtype=np.float32)
        np.add.at(h_out, dst, msg)
        
        out = h_out.reshape(N, self.heads * self.out_features) + self.bias
        return elu(out)

class SnapshotGATEncoder:
    def __init__(self, in_dim=6, hidden_dim=32, out_dim=32, heads=2):
        self.gat1 = SpatialGATLayerNumPy(in_dim, hidden_dim, heads=heads, seed=42)
        self.gat2 = SpatialGATLayerNumPy(hidden_dim * heads, out_dim, heads=1, seed=43)

    def forward(self, x, edge_index):
        h = self.gat1.forward(x, edge_index)
        h = self.gat2.forward(h, edge_index)
        return h
