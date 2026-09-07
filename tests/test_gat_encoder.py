"""
Unit Tests for Multi-Head Spatial Graph Attention (GAT) Snapshot Encoder
Tests feature transformation, attention coefficient computation, multi-head aggregation,
and output embedding dimensions.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gat_encoder import SnapshotGATEncoder, SpatialGATLayerNumPy


class TestSnapshotGATEncoder:
    """Test suite for SnapshotGATEncoder and SpatialGATLayerNumPy."""

    @pytest.fixture
    def encoder(self):
        """Standard 2-head GAT snapshot encoder."""
        return SnapshotGATEncoder(in_dim=6, hidden_dim=32, out_dim=32, heads=2)

    @pytest.fixture
    def sample_graph(self):
        """Synthetic graph with 10 nodes and a set of directed edges."""
        np.random.seed(42)
        num_nodes = 10
        in_dim = 6
        x = np.random.randn(num_nodes, in_dim).astype(np.float32)
        src = list(range(num_nodes)) + [0, 2, 4]
        dst = list(range(1, num_nodes)) + [0, 5, 7, 8]
        edge_index = np.array([src, dst], dtype=np.int64)
        return x, edge_index

    def test_output_shape(self, encoder, sample_graph):
        """Output embedding must match [num_nodes, out_dim]."""
        x, edge_index = sample_graph
        out = encoder.forward(x, edge_index)
        assert out.shape == (x.shape[0], 32), f"Expected {(x.shape[0], 32)}, got {out.shape}"

    def test_output_finite(self, encoder, sample_graph):
        """All output embeddings must be finite real numbers (no NaNs or Infs)."""
        x, edge_index = sample_graph
        out = encoder.forward(x, edge_index)
        assert np.isfinite(out).all(), "Output contains NaN or Inf values"

    def test_single_layer_gat(self):
        """Test individual SpatialGATLayerNumPy forward mechanism."""
        layer = SpatialGATLayerNumPy(in_features=6, out_features=16, heads=2)
        x = np.random.randn(5, 6).astype(np.float32)
        edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        h = layer.forward(x, edge_index)
        assert h.shape == (5, 32)  # heads * out_features = 2 * 16
        assert np.isfinite(h).all()
