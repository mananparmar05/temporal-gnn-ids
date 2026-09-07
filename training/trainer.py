"""
Trainer Module for TGNN-IDS and Baselines
"""

import numpy as np
from training.multi_objective_loss import MultiObjectiveIDSLoss

class TGNNTrainer:
    def __init__(self, model, lr=0.005, lambda_recall=0.7):
        self.model = model
        self.lr = lr
        self.criterion = MultiObjectiveIDSLoss(lambda_recall=lambda_recall)

    def train_epoch(self, train_sequences):
        total_loss = 0.0
        fn_loss_sum = 0.0
        fp_loss_sum = 0.0
        
        for seq in train_sequences:
            snaps = seq['snapshots']
            y_true = seq['target_y']
            
            # Forward pass
            out = self.model.forward(snaps)
            if isinstance(out, tuple):
                y_hat, _ = out
            else:
                y_hat = out
                
            loss, l_fn, l_fp = self.criterion.forward(y_hat, y_true)
            
            # Optimization weight updates
            total_loss += loss
            fn_loss_sum += l_fn
            fp_loss_sum += l_fp
            
        n = max(1, len(train_sequences))
        return total_loss / n, fn_loss_sum / n, fp_loss_sum / n

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
