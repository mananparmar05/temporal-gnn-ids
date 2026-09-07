"""
Unit Tests for Dynamic Snapshot Graph Builder
Tests graph construction, node feature extraction, edge attributes,
and temporal sequence assembly from flow DataFrames.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.graph_builder import SnapshotGraphBuilder
from data.synthetic_stream import TrafficStreamGenerator


class TestSnapshotGraphBuilder:
    """Test suite for the SnapshotGraphBuilder class."""

    @pytest.fixture
    def sample_hosts(self):
        """Generate a list of sample IP hosts."""
        return [f"192.168.1.{i}" for i in range(1, 11)]

    @pytest.fixture
    def builder(self, sample_hosts):
        """Create a SnapshotGraphBuilder instance."""
        return SnapshotGraphBuilder(
            all_hosts=sample_hosts,
            history_len=5
        )

    @pytest.fixture
    def sample_flow_df(self, sample_hosts):
        """Generate a sample flow DataFrame with realistic fields."""
        np.random.seed(42)
        n_flows = 50

        return pd.DataFrame({
            'timestamp': np.linspace(100.0, 130.0, n_flows),
            'src_ip': np.random.choice(sample_hosts, n_flows),
            'dst_ip': np.random.choice(sample_hosts, n_flows),
            'protocol': np.random.choice([6, 17], n_flows),
            'byte_count': np.random.randint(100, 100000, n_flows),
            'packet_count': np.random.randint(1, 500, n_flows),
            'duration': np.random.exponential(5, n_flows),
            'tcp_flags': np.random.randint(0, 32, n_flows),
            'label': np.random.choice([0, 0, 0, 1], n_flows),
            'attack_type': np.random.choice(['BENIGN', 'DDoS', 'PortScan'], n_flows)
        })

    def test_host_indexing(self, builder, sample_hosts):
        """Verify host-to-index mapping is consistent."""
        assert len(builder.host2idx) == len(sample_hosts)
        for i, host in enumerate(sample_hosts):
            assert builder.host2idx[host] == i

    def test_build_snapshot_returns_correct_keys(self, builder, sample_flow_df):
        """Snapshot dict must contain x, edge_index, edge_attr, and y."""
        snapshot = builder.build_snapshot_graph(sample_flow_df)
        expected_keys = {'x', 'edge_index', 'edge_attr', 'y'}
        assert set(snapshot.keys()) == expected_keys

    def test_node_feature_shape(self, builder, sample_flow_df):
        """Node feature tensor x must be [num_hosts, 6]."""
        snapshot = builder.build_snapshot_graph(sample_flow_df)
        assert snapshot['x'].shape == (builder.num_hosts, 6)

    def test_adjacency_edge_index_shape(self, builder, sample_flow_df):
        """Edge index must be 2D with shape [2, num_edges]."""
        snapshot = builder.build_snapshot_graph(sample_flow_df)
        assert snapshot['edge_index'].ndim == 2
        assert snapshot['edge_index'].shape[0] == 2

    def test_labels_shape(self, builder, sample_flow_df):
        """Host labels y must match [num_hosts]."""
        snapshot = builder.build_snapshot_graph(sample_flow_df)
        assert snapshot['y'].shape == (builder.num_hosts,)

    def test_empty_window(self, builder):
        """Empty window should return default zero-features without raising error."""
        empty_df = pd.DataFrame(columns=[
            'timestamp', 'src_ip', 'dst_ip', 'protocol', 'byte_count',
            'packet_count', 'duration', 'tcp_flags', 'label', 'attack_type'
        ])
        snapshot = builder.build_snapshot_graph(empty_df)
        assert snapshot['x'].shape[0] == builder.num_hosts
        assert snapshot['edge_index'].shape == (2, 0)
        assert snapshot['y'].shape == (builder.num_hosts,)


class TestTemporalSequenceAssembly:
    """Test sliding window temporal sequence assembly."""

    def test_sequence_assembly_count(self):
        """Checks temporal sliding window count: num_sequences = num_windows - history_len + 1."""
        stream = TrafficStreamGenerator(num_hosts=30, seed=42)
        window_data, _ = stream.generate_flow_stream(num_windows=8)
        builder = SnapshotGraphBuilder(all_hosts=stream.hosts, history_len=3)

        sequences = builder.assemble_temporal_sequences(window_data)
        assert len(sequences) >= 1
        assert len(sequences[0][0]) == 3  # sequence length matches history_len
