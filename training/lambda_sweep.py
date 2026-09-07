"""
Lambda-Sweep Trainer - Contribution C2
Sweeps lambda_recall / lambda_fp weights to trace the Precision/Recall and FPR Pareto frontier.
"""

import numpy as np
from models.tgnn_ids import TGNN_IDS
from training.trainer import TGNNTrainer

def run_lambda_sweep(train_seqs, val_seqs, lambda_list=[0.1, 0.3, 0.5, 0.7, 0.9], epochs=10):
    """
    Runs model training for each lambda value and collects Recall, FPR, F1 metrics to plot Pareto frontier.
    """
    pareto_results = []
    
    for lam in lambda_list:
        model = TGNN_IDS(node_in_dim=6, hidden_dim=32, history_len=5)
        trainer = TGNNTrainer(model, lr=0.005, lambda_recall=lam)
        
        for epoch in range(epochs):
            trainer.train_epoch(train_seqs)
            
        metrics = trainer.evaluate(val_seqs)
        pareto_results.append({
            'lambda_recall': lam,
            'lambda_fp': 1.0 - lam,
            'recall': metrics['recall'],
            'fpr': metrics['fpr'],
            'precision': metrics['precision'],
            'f1': metrics['f1']
        })
        
    return pareto_results

if __name__ == '__main__':
    print("Lambda sweep module initialized.")
