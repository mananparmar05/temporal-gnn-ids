"""
Multi-Objective Loss Module - Contribution C2
Section 6.5 Implementation:
Loss: L = lambda_recall * L_FN(y_hat, y) + lambda_fp * L_FP(y_hat, y)
where lambda_recall + lambda_fp = 1.
"""

import numpy as np

class MultiObjectiveIDSLoss:
    def __init__(self, lambda_recall=0.7, fn_weight=3.0, eps=1e-7):
        self.lambda_recall = lambda_recall
        self.lambda_fp = 1.0 - lambda_recall
        self.fn_weight = fn_weight
        self.eps = eps

    def forward(self, y_hat, y):
        """
        y_hat: predicted probabilities (N,)
        y: ground truth labels (N,)
        """
        y_hat = np.clip(y_hat, self.eps, 1.0 - self.eps)
        
        attack_mask = (y == 1.0)
        benign_mask = (y == 0.0)
        
        if np.sum(attack_mask) > 0:
            l_fn = -np.mean(self.fn_weight * np.log(y_hat[attack_mask]))
        else:
            l_fn = 0.0
            
        if np.sum(benign_mask) > 0:
            l_fp = -np.mean(np.log(1.0 - y_hat[benign_mask]))
        else:
            l_fp = 0.0
            
        total_loss = self.lambda_recall * l_fn + self.lambda_fp * l_fp
        return total_loss, float(l_fn), float(l_fp)
