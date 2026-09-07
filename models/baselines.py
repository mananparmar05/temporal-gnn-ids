"""
Ablation & Benchmark Baselines (Section 7 & 8)
1. Static-GNN: Graph-aware single snapshot GAT (no temporal attention component)
2. LSTM-only: Temporal sequence model over per-host aggregate features (no graph topology)
3. RandomForestBaseline: Non-graph, non-temporal tabular gradient-boosted / random forest classifier
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.gat_encoder import SnapshotGATEncoder
from models.tgnn_ids import ScoringHeadNumPy

class StaticGNNBaseline:
    """
    Graph-aware baseline with NO temporal component.
    Processes only current snapshot G(t) to isolate Contribution C1.
    """
    def __init__(self, node_in_dim=6, hidden_dim=32):
        self.spatial_encoder = SnapshotGATEncoder(in_dim=node_in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.scoring_head = ScoringHeadNumPy(in_dim=hidden_dim, hidden_dim=16, seed=46)

    def forward(self, current_snapshot):
        x = current_snapshot['x']
        edge_index = current_snapshot['edge_index']
        h_t = self.spatial_encoder.forward(x, edge_index)
        y_hat = self.scoring_head.forward(h_t)
        return y_hat

class LSTMOnlyBaseline:
    """
    Temporal-aware baseline with NO graph structure.
    Processes sequences of per-host node features X(t-K+1)...X(t) with sequence aggregation.
    """
    def __init__(self, node_in_dim=6, hidden_dim=32, history_len=5):
        self.history_len = history_len
        np.random.seed(47)
        self.W_seq = np.random.uniform(-0.1, 0.1, size=(node_in_dim * history_len, hidden_dim)).astype(np.float32)
        self.scoring_head = ScoringHeadNumPy(in_dim=hidden_dim, hidden_dim=16, seed=48)

    def forward(self, sequence_snapshots):
        # Flatten time window sequence for each host
        x_seq_list = [snap['x'] for snap in sequence_snapshots] # List of K (N, in_dim)
        x_flat = np.concatenate(x_seq_list, axis=1) # (N, K * in_dim)
        
        h_seq = np.tanh(np.dot(x_flat, self.W_seq)) # (N, hidden_dim)
        y_hat = self.scoring_head.forward(h_seq)
        return y_hat

class TabularRandomForestBaseline:
    """
    Classical machine learning baseline: Random Forest trained on per-host flow features.
    """
    def __init__(self, n_estimators=50):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict_proba(self, X_test):
        return self.clf.predict_proba(X_test)[:, 1]
