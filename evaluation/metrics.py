"""
Evaluation Metrics Module for TGNN-IDS
Comprehensive metric computation for intrusion detection systems,
including standard classification metrics and IDS-specific measures.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, classification_report
)


class IDSMetrics:
    """
    Centralized metrics calculator for Intrusion Detection Systems.
    
    Computes:
        - Precision, Recall, F1-Score (per-class and macro)
        - False Positive Rate (FPR) — critical for SOC alert fatigue
        - True Positive Rate (TPR / Detection Rate)
        - AUC-ROC and AUC-PR curves
        - Confusion matrix components (TP, FP, TN, FN)
    """
    
    def __init__(self, threshold: float = 0.5, pos_label: int = 1):
        """
        Args:
            threshold: Decision boundary for binary classification.
            pos_label: Label index for the positive (anomaly) class.
        """
        self.threshold = threshold
        self.pos_label = pos_label
    
    def compute_all(
        self, 
        y_true: np.ndarray, 
        y_pred_probs: np.ndarray,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all IDS evaluation metrics.
        
        Args:
            y_true: Ground truth labels [N].
            y_pred_probs: Predicted probabilities for positive class [N].
            y_pred: Hard predictions [N]. If None, derived from y_pred_probs.
        
        Returns:
            Dictionary containing all computed metrics.
        """
        if y_pred is None:
            y_pred = (y_pred_probs >= self.threshold).astype(int)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        
        # Core metrics
        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            
            # IDS-specific metrics
            "detection_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            
            # Confusion matrix raw counts
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }
        
        # AUC metrics (require probability scores)
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred_probs)
        except ValueError:
            metrics["auc_roc"] = 0.0
        
        try:
            metrics["auc_pr"] = average_precision_score(y_true, y_pred_probs)
        except ValueError:
            metrics["auc_pr"] = 0.0
        
        return metrics
    
    def compute_per_attack_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        attack_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-attack-category metrics for multi-class breakdown.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            attack_labels: String labels for attack types.
        
        Returns:
            Nested dict: {attack_type: {precision, recall, f1, count}}
        """
        results = {}
        unique_attacks = np.unique(attack_labels)
        
        for attack in unique_attacks:
            mask = (attack_labels == attack)
            if mask.sum() == 0:
                continue
            
            results[str(attack)] = {
                "precision": float(precision_score(y_true[mask], y_pred[mask], zero_division=0)),
                "recall": float(recall_score(y_true[mask], y_pred[mask], zero_division=0)),
                "f1_score": float(f1_score(y_true[mask], y_pred[mask], zero_division=0)),
                "count": int(mask.sum())
            }
        
        return results
    
    def get_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        optimize_for: str = "f1"
    ) -> Tuple[float, float]:
        """
        Find the optimal classification threshold.
        
        Args:
            y_true: Ground truth labels.
            y_pred_probs: Predicted probabilities.
            optimize_for: Metric to maximize ("f1", "recall", "specificity").
        
        Returns:
            Tuple of (optimal_threshold, best_metric_value).
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)
        
        if optimize_for == "f1":
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            return float(best_threshold), float(f1_scores[best_idx])
        elif optimize_for == "recall":
            # Find threshold that gives recall >= 0.95 with maximum precision
            valid = recalls >= 0.95
            if valid.any():
                best_idx = np.argmax(precisions[valid])
                return float(thresholds[np.where(valid)[0][best_idx]]), float(recalls[np.where(valid)[0][best_idx]])
            return 0.3, float(recalls[0])
        
        return 0.5, 0.0
    
    @staticmethod
    def format_report(metrics: Dict[str, float], title: str = "Evaluation Results") -> str:
        """Format metrics dictionary as a readable report string."""
        lines = [
            f"\n{'='*60}",
            f"  {title}",
            f"{'='*60}",
            f"  Precision:           {metrics.get('precision', 0):.4f}",
            f"  Recall (TPR):        {metrics.get('recall', 0):.4f}",
            f"  F1-Score:            {metrics.get('f1_score', 0):.4f}",
            f"  Accuracy:            {metrics.get('accuracy', 0):.4f}",
            f"  False Positive Rate: {metrics.get('false_positive_rate', 0):.4f}",
            f"  AUC-ROC:             {metrics.get('auc_roc', 0):.4f}",
            f"  AUC-PR:              {metrics.get('auc_pr', 0):.4f}",
            f"{'─'*60}",
            f"  TP: {metrics.get('true_positives', 0):>6d}  |  FP: {metrics.get('false_positives', 0):>6d}",
            f"  FN: {metrics.get('false_negatives', 0):>6d}  |  TN: {metrics.get('true_negatives', 0):>6d}",
            f"{'='*60}\n"
        ]
        return "\n".join(lines)
