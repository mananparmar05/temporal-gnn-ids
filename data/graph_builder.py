"""
Dynamic Snapshot Graph Builder G(t) = (V, E_t, X_t)
Slices continuous traffic flow streams into dynamic graph snapshots and
constructs temporal snapshot sequences for Spatio-Temporal GNN processing.
"""

import numpy as np
from data.parser import FlowLogParser


class SnapshotGraphBuilder:
    def __init__(self, all_hosts, history_len=5, feature_dim=6):
        self.all_hosts = list(all_hosts)
        self.host2idx = {host: idx for idx, host in enumerate(self.all_hosts)}
        self.num_hosts = len(self.all_hosts)
        self.history_len = history_len
        self.feature_dim = feature_dim
        self.parser = FlowLogParser()

    def build_snapshot_graph(self, window_df):
        """
        Builds graph snapshot G(t) from a window DataFrame.
        """
        N = self.num_hosts
        node_stats = {i: {'in_deg': 0, 'out_deg': 0, 'bytes_sent': 0.0,
                          'bytes_recv': 0.0, 'pkts_sent': 0.0, 'duration_sum': 0.0}
                      for i in range(N)}

        src_indices = []
        dst_indices = []
        edge_attrs = []

        if len(window_df) > 0:
            norm_edge_feats = self.parser.normalize_features(window_df)

            for i, (_, row) in enumerate(window_df.iterrows()):
                src_ip = row['src_ip']
                dst_ip = row['dst_ip']

                if src_ip in self.host2idx and dst_ip in self.host2idx:
                    u = self.host2idx[src_ip]
                    v = self.host2idx[dst_ip]

                    src_indices.append(u)
                    dst_indices.append(v)
                    edge_attrs.append(norm_edge_feats[i])

                    # Accumulate node statistics
                    node_stats[u]['out_deg'] += 1
                    node_stats[v]['in_deg'] += 1

                    b = float(row.get('byte_count', 0))
                    p = float(row.get('packet_count', 0))
                    d = float(row.get('duration', 0))

                    node_stats[u]['bytes_sent'] += b
                    node_stats[v]['bytes_recv'] += b
                    node_stats[u]['pkts_sent'] += p
                    node_stats[u]['duration_sum'] += d

        # Construct Node Feature Matrix X(t) [N x 6]
        X = np.zeros((N, self.feature_dim), dtype=np.float32)
        for i in range(N):
            st = node_stats[i]
            X[i] = [
                np.log1p(st['in_deg']),
                np.log1p(st['out_deg']),
                np.log1p(st['bytes_sent']),
                np.log1p(st['bytes_recv']),
                np.log1p(st['pkts_sent']),
                np.log1p(st['duration_sum'])
            ]

        # Adjacency Edge Index [2 x E]
        if len(src_indices) > 0:
            edge_index = np.array([src_indices, dst_indices], dtype=np.int64)
            edge_attr = np.array(edge_attrs, dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 5), dtype=np.float32)

        # Host Labels Y(t) [N] (1 if host was involved in any attack in window, else 0)
        y = np.zeros(N, dtype=np.float32)
        if 'label' in window_df.columns and len(window_df) > 0:
            for _, row in window_df.iterrows():
                if row.get('label', 0) == 1:
                    src_ip = row['src_ip']
                    dst_ip = row['dst_ip']
                    if src_ip in self.host2idx:
                        y[self.host2idx[src_ip]] = 1.0
                    if dst_ip in self.host2idx:
                        y[self.host2idx[dst_ip]] = 1.0

        return {
            'x': X,
            'node_features': X,       # Alias
            'edge_index': edge_index,
            'adjacency': edge_index,   # Alias
            'edge_attr': edge_attr,
            'y': y,
            'labels': y                # Alias
        }

    def assemble_temporal_sequences(self, window_streams):
        """
        Assembles temporal sliding sequences of length K.
        Returns:
            seqs: list of dicts with 'snapshots' and 'target_y'
            targets: list of target_y arrays
        """
        snapshots = [self.build_snapshot_graph(w['df'] if isinstance(w, dict) else w) for w in window_streams]

        seqs = []
        targets = []

        for i in range(len(snapshots) - self.history_len + 1):
            sub_snaps = snapshots[i:i + self.history_len]
            target_y = sub_snaps[-1]['y']
            seqs.append({
                'time_step': i + self.history_len - 1,
                'snapshots': sub_snaps,
                'target_y': target_y
            })
            targets.append(target_y)

        return seqs, targets
