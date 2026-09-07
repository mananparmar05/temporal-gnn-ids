"""
Logging Module for TGNN-IDS
Provides structured, color-coded logging with file and console output.
Supports per-experiment log files and metric tracking.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path


class MetricTracker:
    """
    Tracks training metrics across epochs for visualization and checkpointing.
    Records loss, precision, recall, F1, FPR, and custom metrics per epoch.
    """
    
    def __init__(self):
        self.history = {}
    
    def update(self, epoch: int, metrics: dict) -> None:
        """Record metrics for a given epoch."""
        self.history[epoch] = {
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
    
    def get_best(self, metric: str, mode: str = "max") -> dict:
        """
        Find the epoch with the best value for a given metric.
        
        Args:
            metric: Name of the metric to optimize.
            mode: "max" for metrics like F1/Recall, "min" for loss.
        
        Returns:
            dict with 'epoch' and 'value' keys.
        """
        if not self.history:
            return {"epoch": -1, "value": None}
        
        compare_fn = max if mode == "max" else min
        best_epoch = compare_fn(
            self.history.keys(),
            key=lambda e: self.history[e].get(metric, float('-inf') if mode == "max" else float('inf'))
        )
        return {
            "epoch": best_epoch,
            "value": self.history[best_epoch].get(metric)
        }
    
    def save(self, filepath: str) -> None:
        """Save complete metric history to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load metric history from JSON file."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)


def setup_logger(
    name: str = "tgnn_ids",
    log_dir: str = "results/training_logs",
    level: str = "INFO",
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Configure and return a structured logger with console and file handlers.
    
    Args:
        name: Logger name identifier.
        log_dir: Directory for log file output.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Enable colored console output.
        file_logging: Enable persistent file logging.
    
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    # Formatter
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler with color support
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)
    
    # File handler for persistent logging
    if file_logging:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger
