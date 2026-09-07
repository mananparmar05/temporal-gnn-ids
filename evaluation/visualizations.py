"""
Visualization Module for TGNN-IDS Evaluation
Generates publication-quality plots for model analysis and reporting.

Supported visualizations:
  - Training loss curves (FN loss, FP loss, total loss)
  - Confusion matrix heatmap
  - ROC and Precision-Recall curves
  - Pareto frontier (Precision vs Recall at different lambda values)
  - Temporal attention weight heatmaps
  - Per-attack-type performance bar charts
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from typing import Dict, List, Optional, Tuple


# Publication-quality plot defaults
plt.rcParams.update({
    'figure.figsize': (10, 7),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

PALETTE = sns.color_palette("husl", 8)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = "results/figures/training_curves.png"
) -> str:
    """
    Plot training and validation loss curves across epochs.
    
    Args:
        history: Dict with keys like 'train_loss', 'val_loss', 'fn_loss', 'fp_loss'.
        save_path: Output file path.
    
    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss curves
    ax1 = axes[0]
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], label='Train Loss', color=PALETTE[0], linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Val Loss', color=PALETTE[1], linewidth=2, linestyle='--')
    if 'fn_loss' in history:
        ax1.plot(epochs, history['fn_loss'], label='FN Loss (Missed)', color=PALETTE[2], linewidth=1.5, alpha=0.7)
    if 'fp_loss' in history:
        ax1.plot(epochs, history['fp_loss'], label='FP Loss (False Alerts)', color=PALETTE[3], linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 / Recall curves
    ax2 = axes[1]
    if 'val_f1' in history:
        ax2.plot(epochs, history['val_f1'], label='Val F1', color=PALETTE[4], linewidth=2)
    if 'val_recall' in history:
        ax2.plot(epochs, history['val_recall'], label='Val Recall', color=PALETTE[5], linewidth=2)
    if 'val_precision' in history:
        ax2.plot(epochs, history['val_precision'], label='Val Precision', color=PALETTE[6], linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.suptitle('TGNN-IDS Training Progress', fontsize=16, fontweight='bold', y=1.02)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return save_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ["Benign", "Anomaly"],
    save_path: str = "results/figures/confusion_matrix.png",
    normalize: bool = True
) -> str:
    """
    Plot confusion matrix heatmap with counts and percentages.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class names for display.
        save_path: Output path.
        normalize: If True, shows percentages alongside raw counts.
    
    Returns:
        Path to saved figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        annot = np.array([
            [f"{cm[i,j]}\n({cm_norm[i,j]:.1%})" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])
        sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues', 
                     xticklabels=labels, yticklabels=labels, ax=ax,
                     vmin=0, vmax=1, linewidths=0.5)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, ax=ax,
                     linewidths=0.5)
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('TGNN-IDS Confusion Matrix', fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return save_path


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    model_name: str = "TGNN-IDS",
    save_path: str = "results/figures/roc_pr_curves.png"
) -> str:
    """
    Plot ROC curve and Precision-Recall curve side by side.
    
    Args:
        y_true: Ground truth labels.
        y_pred_probs: Predicted probabilities for positive class.
        model_name: Model identifier for the legend.
        save_path: Output path.
    
    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color=PALETTE[0], linewidth=2.5,
                  label=f'{model_name} (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    axes[0].fill_between(fpr, tpr, alpha=0.15, color=PALETTE[0])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall, precision)
    
    axes[1].plot(recall, precision, color=PALETTE[1], linewidth=2.5,
                  label=f'{model_name} (AUC = {pr_auc:.4f})')
    axes[1].fill_between(recall, precision, alpha=0.15, color=PALETTE[1])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} — Detection Performance', fontsize=16, fontweight='bold', y=1.02)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return save_path


def plot_pareto_frontier(
    sweep_results: List[Dict[str, float]],
    save_path: str = "results/figures/pareto_frontier.png"
) -> str:
    """
    Plot the Pareto frontier showing Precision vs Recall tradeoff
    across different lambda_recall values.
    
    Args:
        sweep_results: List of dicts with 'lambda_recall', 'precision', 'recall', 'fpr'.
        save_path: Output path.
    
    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    lambdas = [r['lambda_recall'] for r in sweep_results]
    precisions = [r['precision'] for r in sweep_results]
    recalls = [r['recall'] for r in sweep_results]
    fprs = [r.get('fpr', 0) for r in sweep_results]
    
    # Precision vs Recall Pareto
    scatter = axes[0].scatter(recalls, precisions, c=lambdas, cmap='viridis', 
                               s=100, edgecolors='black', linewidth=0.5, zorder=5)
    axes[0].plot(recalls, precisions, '--', color='gray', alpha=0.5, zorder=1)
    axes[0].set_xlabel('Recall (Detection Rate)')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Pareto Frontier')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='lambda_recall')
    
    # FPR vs Recall
    scatter2 = axes[1].scatter(recalls, fprs, c=lambdas, cmap='viridis',
                                s=100, edgecolors='black', linewidth=0.5, zorder=5)
    axes[1].plot(recalls, fprs, '--', color='gray', alpha=0.5, zorder=1)
    axes[1].set_xlabel('Recall (Detection Rate)')
    axes[1].set_ylabel('False Positive Rate')
    axes[1].set_title('FPR vs Detection Rate Tradeoff')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='lambda_recall')
    
    plt.suptitle('Lambda-Sweep Pareto Frontier Analysis', fontsize=16, fontweight='bold', y=1.02)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return save_path


def plot_temporal_attention_heatmap(
    attention_weights: np.ndarray,
    window_labels: Optional[List[str]] = None,
    node_labels: Optional[List[str]] = None,
    save_path: str = "results/figures/temporal_attention.png"
) -> str:
    """
    Plot temporal attention weight heatmap for XAI explainability.
    Shows which historical time windows the model attends to most.
    
    Args:
        attention_weights: Attention matrix [num_heads, K, K] or [K, K].
        window_labels: Labels for each time window.
        node_labels: Labels for nodes (if node-level attention).
        save_path: Output path.
    
    Returns:
        Path to saved figure.
    """
    if attention_weights.ndim == 3:
        # Average across attention heads
        attention_weights = attention_weights.mean(axis=0)
    
    K = attention_weights.shape[0]
    
    if window_labels is None:
        window_labels = [f"t-{K-1-i}" if i < K-1 else "t (current)" for i in range(K)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        attention_weights,
        annot=True, fmt='.3f',
        cmap='YlOrRd',
        xticklabels=window_labels,
        yticklabels=window_labels,
        ax=ax,
        linewidths=0.5,
        vmin=0
    )
    
    ax.set_xlabel('Key (Historical Windows)', fontweight='bold')
    ax.set_ylabel('Query (Current Windows)', fontweight='bold')
    ax.set_title('Temporal Self-Attention Weights\n'
                  '(Which historical windows contribute to anomaly detection)',
                  fontsize=13, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return save_path
