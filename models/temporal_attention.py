"""
Temporal Graph Attention Module — Contribution C1
Section 6.3 Implementation:
Learns which past windows (t-k) and neighbor states matter most to node i's current anomaly score:
\beta_k(t) = softmax_k( f(h_i(t-k), h_i(t)) )
z_i(t) = \sum_{k=0}^{K-1} \beta_k(t) \cdot h_i(t-k)
"""

import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class TemporalGraphAttention:
    def __init__(self, embed_dim=32, history_len=5, seed=44):
        np.random.seed(seed)
        self.embed_dim = embed_dim
        self.history_len = history_len
        
        limit = np.sqrt(6.0 / (embed_dim + embed_dim))
        self.W_temp = np.random.uniform(-limit, limit, size=(embed_dim, embed_dim)).astype(np.float32)
        self.U_temp = np.random.uniform(-limit, limit, size=(embed_dim, embed_dim)).astype(np.float32)
        self.v_temp = np.random.uniform(-limit, limit, size=(embed_dim, 1)).astype(np.float32)

    def forward(self, h_seq):
        """
        h_seq: List of K snapshot embeddings [h(t-K+1), ..., h(t)], each of shape (N, embed_dim)
        Returns:
            z: Temporal context representation of shape (N, embed_dim)
            beta: Attention weight matrix of shape (N, K) showing attribution to each past window.
        """
        K = len(h_seq)
        N = h_seq[0].shape[0]
        
        h_current = h_seq[-1] # h(t)
        
        scores = []
        for h_past in h_seq:
            # f(h_i(t-k), h_i(t)) = v^T * tanh( W * h(t-k) + U * h(t) )
            proj = np.tanh(np.dot(h_past, self.W_temp) + np.dot(h_current, self.U_temp)) # (N, embed_dim)
            score = np.dot(proj, self.v_temp).squeeze(-1) # (N,)
            scores.append(score)
            
        scores = np.stack(scores, axis=1) # (N, K)
        beta = softmax(scores, axis=1) # (N, K)
        
        z = np.zeros_like(h_current)
        for k_idx in range(K):
            z += beta[:, k_idx, np.newaxis] * h_seq[k_idx]
            
        return z, beta
