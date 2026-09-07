"""
Early Stopping Module for TGNN-IDS Training
Monitors a validation metric and halts training when no improvement
is observed for a specified number of consecutive epochs (patience).

Supports monitoring any metric (val_loss, val_f1, val_recall) and 
automatically saves the best model checkpoint.
"""

import os
import torch
import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early Stopping callback to prevent overfitting.
    
    Tracks a monitored metric across epochs and triggers a stop signal
    when the metric fails to improve by at least `min_delta` for 
    `patience` consecutive epochs.
    
    Usage:
        early_stop = EarlyStopping(patience=15, monitor='val_f1', mode='max')
        
        for epoch in range(max_epochs):
            val_metrics = train_one_epoch(...)
            
            if early_stop(val_metrics['val_f1'], model):
                print("Early stopping triggered!")
                break
        
        # Load best weights
        early_stop.load_best_weights(model)
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        monitor: str = "val_f1",
        mode: str = "max",
        checkpoint_dir: str = "results/checkpoints",
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change in monitored metric to qualify as improvement.
            monitor: Name of the metric to track.
            mode: "max" if higher is better (F1, Recall), "min" if lower is better (loss).
            checkpoint_dir: Directory to save best model weights.
            verbose: Print status messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = -1
        self.should_stop = False
        self._checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        
        if mode == "max":
            self._is_better = lambda current, best: current > best + min_delta
        else:
            self._is_better = lambda current, best: current < best - min_delta
    
    def __call__(self, current_value: float, model: torch.nn.Module, epoch: int = -1) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current epoch's monitored metric value.
            model: Model to checkpoint if this is the best epoch.
            epoch: Current epoch number (for logging).
        
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_value
            self.best_epoch = epoch
            self._save_checkpoint(model, epoch, current_value)
            return False
        
        if self._is_better(current_value, self.best_score):
            if self.verbose:
                improvement = current_value - self.best_score
                print(f"  [EarlyStopping] {self.monitor} improved by {improvement:+.4f} "
                      f"({self.best_score:.4f} → {current_value:.4f}). Saving checkpoint.")
            
            self.best_score = current_value
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model, epoch, current_value)
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement in {self.monitor} for "
                      f"{self.counter}/{self.patience} epochs.")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] *** STOPPING *** Best {self.monitor}: "
                          f"{self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        
        return False
    
    def _save_checkpoint(self, model: torch.nn.Module, epoch: int, score: float) -> None:
        """Save model state dict to checkpoint file."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            f'best_{self.monitor}': score
        }, self._checkpoint_path)
    
    def load_best_weights(self, model: torch.nn.Module) -> None:
        """Load the best model weights from checkpoint."""
        if os.path.exists(self._checkpoint_path):
            checkpoint = torch.load(self._checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  [EarlyStopping] Loaded best weights from epoch {checkpoint['epoch']}")
        else:
            print("  [EarlyStopping] No checkpoint found. Using current weights.")
