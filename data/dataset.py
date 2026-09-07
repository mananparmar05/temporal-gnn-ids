"""
PyTorch Dataset & DataLoader Wrappers for TGNN-IDS
Provides efficient batched data loading for temporal graph snapshot sequences.
Supports both CIC-IDS2017 and UNSW-NB15 benchmark datasets with
on-the-fly graph construction and caching.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from data.graph_builder import SnapshotGraphBuilder


class TemporalGraphDataset(Dataset):
    """
    PyTorch Dataset for Temporal Graph Snapshot Sequences.
    
    Each sample is a sequence of K consecutive snapshot graphs:
        [G(t-K+1), G(t-K+2), ..., G(t)]
    
    where each G(t) contains:
        - node_features: [num_nodes, feature_dim] tensor
        - adjacency:     [num_nodes, num_nodes] sparse/dense tensor
        - edge_features: [num_edges, edge_dim] tensor
        - labels:        [num_nodes] binary labels (0=benign, 1=anomaly)
    """
    
    def __init__(
        self,
        flow_windows: List[pd.DataFrame],
        hosts: List[str],
        history_len: int = 5,
        feature_dim: int = 6,
        cache_dir: Optional[str] = None,
        transform=None
    ):
        """
        Args:
            flow_windows: List of DataFrames, one per time window.
            hosts: List of unique IP host addresses across all windows.
            history_len: K - number of consecutive snapshots per sample.
            feature_dim: Dimension of per-node flow feature vectors.
            cache_dir: If set, caches preprocessed tensors to disk.
            transform: Optional transform applied to each sample.
        """
        super().__init__()
        self.flow_windows = flow_windows
        self.hosts = hosts
        self.history_len = history_len
        self.feature_dim = feature_dim
        self.cache_dir = cache_dir
        self.transform = transform
        
        self.graph_builder = SnapshotGraphBuilder(
            all_hosts=hosts,
            history_len=history_len,
            feature_dim=feature_dim
        )
        
        # Build all snapshot graphs upfront
        self._snapshots = self._build_all_snapshots()
        
        # Number of valid temporal sequences
        self.num_sequences = max(0, len(self._snapshots) - history_len + 1)
    
    def _build_all_snapshots(self) -> List[Dict]:
        """Convert all flow windows into graph snapshot dictionaries."""
        snapshots = []
        for window_df in self.flow_windows:
            snap = self.graph_builder.build_snapshot_graph(window_df)
            snapshots.append(snap)
        return snapshots
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a temporal sequence of K snapshot graphs.
        
        Returns:
            dict with keys:
                - 'node_features': [K, num_nodes, feature_dim]
                - 'adjacency':     [K, num_nodes, num_nodes]
                - 'labels':        [num_nodes] (from the final snapshot G(t))
                - 'timestamps':    [K] integer indices
        """
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"seq_{idx}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path)
        
        sequence = self._snapshots[idx:idx + self.history_len]
        
        node_features = torch.stack([
            torch.tensor(s['node_features'], dtype=torch.float32)
            for s in sequence
        ])
        
        adjacency = torch.stack([
            torch.tensor(s['adjacency'], dtype=torch.float32)
            for s in sequence
        ])
        
        # Labels from the last (current) snapshot
        labels = torch.tensor(
            sequence[-1]['labels'], dtype=torch.long
        )
        
        timestamps = torch.arange(idx, idx + self.history_len, dtype=torch.long)
        
        sample = {
            'node_features': node_features,
            'adjacency': adjacency,
            'labels': labels,
            'timestamps': timestamps
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(sample, cache_path)
        
        return sample


def create_dataloaders(
    flow_windows: List[pd.DataFrame],
    hosts: List[str],
    history_len: int = 5,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test DataLoaders with stratified temporal splits.
    
    Args:
        flow_windows: Full list of time-window DataFrames.
        hosts: IP host list.
        history_len: Snapshot sequence length K.
        batch_size: Samples per batch.
        train_split: Fraction for training.
        val_split: Fraction for validation.
        num_workers: Parallel data loading workers.
        pin_memory: Pin tensors in CPU memory for faster GPU transfer.
        seed: Random seed for split reproducibility.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dataset = TemporalGraphDataset(
        flow_windows=flow_windows,
        hosts=hosts,
        history_len=history_len
    )
    
    total = len(dataset)
    n_train = int(total * train_split)
    n_val = int(total * val_split)
    n_test = total - n_train - n_val
    
    # Temporal split (no shuffling — preserves chronological order)
    train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    test_dataset = torch.utils.data.Subset(dataset, range(n_train + n_val, total))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    print(f"[Data] Splits — Train: {n_train} | Val: {n_val} | Test: {n_test}")
    
    return train_loader, val_loader, test_loader
