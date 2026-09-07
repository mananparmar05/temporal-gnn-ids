"""
Snapshot Graph & Temporal Sequence Assembly Module
Builds dynamic snapshot graphs G(t) = (V(t), E(t)) from flow streams
and chains K consecutive snapshots into sliding temporal window sequences G(t-K+1)...G(t).
"""

import numpy as np
from data.parser import FlowLogParser

class SnapshotGraphBuilder:
    def __init__(self, all_hosts, history_len=5, feature_dim=6):
        self.all_hosts = list(all_hosts)
        self.host2idx = {h: i for i, h in enumerate(self.all_hosts)}
        self.num_hosts = len(self.all_hosts)
        self.history_len = history_len
        self.parser = FlowLogParser()

    def build_snapshot_graph(self, window_df):
        """
        Converts a single time-window flow DataFrame into a snapshot graph object.
        Returns node features X(t), edge indices E(t), edge features EdgeAttr(t), and node labels Y(t).
        """
        N = self.num_hosts
        node_stats = {i: {'in_deg': 0, 'out_deg': 0, 'bytes_sent': 0.0, 'bytes_recv': 0.0, 
                          'pkts_sent': 0.0, 'duration_sum': 0.0, 'label': 0} 
                      for i in range(N)}
        
        src_indices = []
        dst_indices = []
        edge_attrs = []
        
        if len(window_df) > 0:
            norm_edge_feats = self.parser.normalize_features(window_df)
            
            for idx, row in window_df.iterrows():
                src_ip = row['src_ip']
                dst_ip = row['dst_ip']
                lbl = row.get('label', 0)
                
                if src_ip in self.host2idx and dst_ip in self.host2idx:
                    u = self.host2idx[src_ip]
                    v = self.host2idx[dst_ip]
                    
                    src_indices.append(u)
                    dst_indices.append(v)
                    edge_attrs.append(norm_edge_feats[idx])
                    
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
                    
                    if lbl == 1:
                        node_stats[u]['label'] = 1
                        node_stats[v]['label'] = 1

        # Construct Node Features Matrix X of shape (N, feature_dim)
        X = np.zeros((N, 6), dtype=np.float32)
        Y = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            s = node_stats[i]
            deg = s['in_deg'] + s['out_deg']
            avg_dur = s['duration_sum'] / max(1, deg)
            
            X[i] = [
                np.log1p(s['in_deg']),
                np.log1p(s['out_deg']),
                np.log1p(s['bytes_sent']),
                np.log1p(s['bytes_recv']),
                np.log1p(s['pkts_sent']),
                np.log1p(avg_dur)
            ]
            Y[i] = s['label']

        if len(src_indices) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 5), dtype=np.float32)
        else:
            edge_index = np.array([src_indices, dst_indices], dtype=np.int64)
            edge_attr = np.array(edge_attrs, dtype=np.float32)
            
        return {
            'x': X,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'y': Y
        }

    def assemble_temporal_sequences(self, window_streams):
        """
        Chains K consecutive snapshots into dynamic temporal sequences.
        Returns sequences of K snapshots and target label y(t) at time step t.
        """
        snapshots = [self.build_snapshot_graph(w['df']) for w in window_streams]
        
        sequences = []
        K = self.history_len
        for t in range(K - 1, len(snapshots)):
            seq_snapshots = snapshots[t - K + 1 : t + 1] # length K
            target_y = snapshots[t]['y']
            sequences.append({
                'time_step': t,
                'snapshots': seq_snapshots,
                'target_y': target_y
            })
            
        return sequences, snapshots
