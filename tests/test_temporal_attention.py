"""
Unit Tests for Temporal Graph Attention Mechanism (Contribution C1)
Tests temporal attention weight computation, multi-head temporal pooling,
interpretability weight (beta_k) extraction, and temporal responsiveness.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.temporal_attention import TemporalGraphAttention


class TestTemporalGraphAttention:
    """Test suite for the TemporalGraphAttention module."""

    @pytest.fixture
    def temporal_attn(self):
        """Construct standard TemporalGraphAttention layer."""
        return TemporalGraphAttention(embed_dim=32, history_len=5, seed=42)

    @pytest.fixture
    def sample_history_embeddings(self):
        """Sequence of K=5 snapshot embeddings for 10 hosts."""
        np.random.seed(42)
        num_nodes = 10
        embed_dim = 32
        history_len = 5
        h_seq = [np.random.randn(num_nodes, embed_dim).astype(np.float32) for _ in range(history_len)]
        return h_seq

    def test_output_shapes(self, temporal_attn, sample_history_embeddings):
        """Output pooled embedding must be [N, embed_dim] and beta weights [N, K]."""
        h_seq = sample_history_embeddings
        num_nodes = h_seq[0].shape[0]
        history_len = len(h_seq)
        embed_dim = h_seq[0].shape[1]

        z, beta = temporal_attn.forward(h_seq)
        assert z.shape == (num_nodes, embed_dim), f"Expected {(num_nodes, embed_dim)}, got {z.shape}"
        assert beta.shape == (num_nodes, history_len), f"Expected {(num_nodes, history_len)}, got {beta.shape}"

    def test_beta_weights_probability_distribution(self, temporal_attn, sample_history_embeddings):
        """Beta attention weights must be valid probabilities: >= 0 and sum to 1.0 per node."""
        h_seq = sample_history_embeddings
        _, beta = temporal_attn.forward(h_seq)

        assert (beta >= 0.0).all(), "Beta weights contain negative values"
        assert (beta <= 1.0).all(), "Beta weights contain values > 1"

        row_sums = beta.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-5, atol=1e-5)

    def test_temporal_sensitivity(self, temporal_attn):
        """Verifies that shifting snapshot embeddings dynamically influences the beta distribution."""
        num_nodes, embed_dim, history_len = 4, 32, 5
        np.random.seed(99)
        h_base = [np.random.randn(num_nodes, embed_dim).astype(np.float32) for _ in range(history_len)]

        h_burst = [h.copy() for h in h_base]
        # Inject significant anomaly burst into latest window t
        h_burst[-1] += 20.0

        _, beta_base = temporal_attn.forward(h_base)
        _, beta_burst = temporal_attn.forward(h_burst)

        diff = np.abs(beta_burst - beta_base).sum()
        assert diff > 0.01, "Attention weights should be sensitive to changes in snapshot embeddings"
