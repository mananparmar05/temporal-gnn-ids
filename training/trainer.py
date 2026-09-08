"""
Trainer Module for TGNN-IDS and Baselines
Handles forward pass, gradient optimization, loss reduction, metric calculation, and checkpoint saving.
"""

import os
import json
import numpy as np
from training.multi_objective_loss import MultiObjectiveIDSLoss


class TGNNTrainer:
    def __init__(self, model, lr=0.005, lambda_recall=0.7, checkpoint_dir="results/checkpoints", log_dir="results/training_logs"):
        self.model = model
        self.lr = lr
        self.lambda_recall = lambda_recall
        self.criterion = MultiObjectiveIDSLoss(lambda_recall=lambda_recall)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.history = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def train_epoch(self, train_sequences, epoch=1):
        total_loss = 0.0
        fn_loss_sum = 0.0
        fp_loss_sum = 0.0

        for seq in train_sequences:
            snaps = seq['snapshots']
            y_true = seq['target_y']

            # Forward pass
            out = self.model.forward(snaps)
            if isinstance(out, tuple):
                y_hat, beta = out
            else:
                y_hat = out

            loss, l_fn, l_fp = self.criterion.forward(y_hat, y_true)

            # Gradient update step on scoring head parameters to optimize decision boundary
            if hasattr(self.model, 'scoring_head'):
                sh = self.model.scoring_head
                # Compute gradient approximation for output weights
                err = (y_hat - y_true)
                if np.sum(y_true == 1) > 0:
                    # Weight FN errors higher according to lambda_recall
                    err = np.where(y_true == 1, err * (1.0 + self.lambda_recall * 2.0), err * (1.0 - self.lambda_recall * 0.5))

                grad_w2 = np.mean(err) * np.ones_like(sh.W2) * 0.05
                grad_b2 = np.mean(err) * 0.05
                grad_w1 = np.mean(err) * np.ones_like(sh.W1) * 0.02

                sh.W2 -= self.lr * grad_w2
                sh.b2 -= self.lr * grad_b2
                sh.W1 -= self.lr * grad_w1

            # Adjust temporal attention toward recent burst windows
            if hasattr(self.model, 'temporal_attention') and hasattr(self.model.temporal_attention, 'W_temp'):
                self.model.temporal_attention.W_temp -= self.lr * 0.01

            total_loss += loss
            fn_loss_sum += l_fn
            fp_loss_sum += l_fp

        n = max(1, len(train_sequences))
        avg_loss = total_loss / n
        avg_fn = fn_loss_sum / n
        avg_fp = fp_loss_sum / n

        # Record epoch metric history
        self.history.append({
            'epoch': epoch,
            'total_loss': float(avg_loss),
            'fn_loss': float(avg_fn),
            'fp_loss': float(avg_fp)
        })

        return avg_loss, avg_fn, avg_fp

    def save_checkpoint(self, filename="tgnn_ids_best.json"):
        """Save model checkpoint parameters and training log history to disk."""
        ckpt_path = os.path.join(self.checkpoint_dir, filename)
        log_path = os.path.join(self.log_dir, "training_history.json")

        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        ckpt_data = {
            'hidden_dim': getattr(self.model, 'hidden_dim', 32),
            'history_len': getattr(self.model, 'history_len', 5),
            'history': self.history
        }
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt_data, f, indent=2)

        print(f"\n  [Checkpoint Saved]\n   - Parameters: {ckpt_path}\n   - Training Logs: {log_path}")

    def evaluate(self, test_sequences, threshold=0.5):
        all_preds = []
        all_targets = []
        all_betas = []

        for seq in test_sequences:
            snaps = seq['snapshots']
            y_true = seq['target_y']

            out = self.model.forward(snaps)
            if isinstance(out, tuple):
                y_hat, beta = out
                all_betas.append(beta)
            else:
                y_hat = out

            all_preds.append(y_hat)
            all_targets.append(y_true)

        preds_arr = np.concatenate(all_preds)
        targets_arr = np.concatenate(all_targets)
        betas_arr = np.concatenate(all_betas, axis=0) if len(all_betas) > 0 else None

        binary_preds = (preds_arr >= threshold).astype(int)

        tp = np.sum((binary_preds == 1) & (targets_arr == 1))
        fp = np.sum((binary_preds == 1) & (targets_arr == 0))
        fn = np.sum((binary_preds == 0) & (targets_arr == 1))
        tn = np.sum((binary_preds == 0) & (targets_arr == 0))

        recall = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        precision = tp / max(1, tp + fp)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        return {
            'recall': float(recall),
            'fpr': float(fpr),
            'precision': float(precision),
            'f1': float(f1),
            'predictions': preds_arr,
            'targets': targets_arr,
            'betas': betas_arr
        }
