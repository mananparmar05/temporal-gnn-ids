"""
Full Temporal Graph Neural Network Intrusion Detector (TGNN-IDS)
Combines Spatial GAT Snapshot Encoding, Temporal Graph Attention (C1), 
and Anomaly Scoring Head (Section 6.4).
"""

import numpy as np
from models.gat_encoder import SnapshotGATEncoder
from models.temporal_attention import TemporalGraphAttention

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def relu(x):
    return np.maximum(0, x)

class ScoringHeadNumPy:
    def __init__(self, in_dim=32, hidden_dim=16, seed=45):
        np.random.seed(seed)
        limit1 = np.sqrt(6.0 / (in_dim + hidden_dim))
        limit2 = np.sqrt(6.0 / (hidden_dim + 1))
        self.W1 = np.random.uniform(-limit1, limit1, size=(in_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.uniform(-limit2, limit2, size=(hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, z):
        h1 = relu(np.dot(z, self.W1) + self.b1)
        out = sigmoid(np.dot(h1, self.W2) + self.b2).squeeze(-1)
        return out

class TGNN_IDS:
    def __init__(self, node_in_dim=6, hidden_dim=32, history_len=5):
        self.history_len = history_len
        self.spatial_encoder = SnapshotGATEncoder(in_dim=node_in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.temporal_attention = TemporalGraphAttention(embed_dim=hidden_dim, history_len=history_len)
        self.scoring_head = ScoringHeadNumPy(in_dim=hidden_dim, hidden_dim=16)

    def forward(self, sequence_snapshots):
        """
        sequence_snapshots: List of K snapshot dicts
        Returns:
            y_hat: (N,) predicted anomaly probability per host at time t
            beta: (N, K) learned temporal attention weights
        """
        h_seq = []
        for snapshot in sequence_snapshots:
            x = snapshot['x']
            edge_index = snapshot['edge_index']
            h_t = self.spatial_encoder.forward(x, edge_index)
            h_seq.append(h_t)
            
        z, beta = self.temporal_attention.forward(h_seq)
        y_hat = self.scoring_head.forward(z)
        
        return y_hat, beta
