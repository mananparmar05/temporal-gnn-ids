"""
Cross-Dataset Zero-Shot Generalization Harness — Contribution C3
Section 6.6 Implementation:
Trains TGNN-IDS on Dataset A attack taxonomy (e.g., CIC-IDS2017) 
and evaluates zero-shot on Dataset B attack taxonomy (e.g., UNSW-NB15).
Calculates cross-dataset recall drop metric (Core C3 metric).
"""

import numpy as np
from models.tgnn_ids import TGNN_IDS
from models.baselines import StaticGNNBaseline, LSTMOnlyBaseline, TabularRandomForestBaseline
from training.trainer import TGNNTrainer

class CrossDatasetEvaluator:
    def __init__(self, dataset_a_train_seqs, dataset_a_test_seqs, dataset_b_zero_shot_seqs):
        self.ds_a_train = dataset_a_train_seqs
        self.ds_a_test = dataset_a_test_seqs
        self.ds_b_test = dataset_b_zero_shot_seqs

    def evaluate_all_models(self, epochs=10):
        results = {}
        
        # 1. Full TGNN-IDS Model (This Project)
        tgnn_model = TGNN_IDS(node_in_dim=6, hidden_dim=32, history_len=5)
        trainer_tgnn = TGNNTrainer(tgnn_model, lr=0.005, lambda_recall=0.7)
        for _ in range(epochs):
            trainer_tgnn.train_epoch(self.ds_a_train)
            
        metrics_a_tgnn = trainer_tgnn.evaluate(self.ds_a_test)
        metrics_b_tgnn = trainer_tgnn.evaluate(self.ds_b_test)
        
        recall_drop_tgnn = metrics_a_tgnn['recall'] - metrics_b_tgnn['recall']
        
        results['TGNN-IDS (This Project)'] = {
            'in_dist_recall': metrics_a_tgnn['recall'],
            'in_dist_f1': metrics_a_tgnn['f1'],
            'in_dist_fpr': metrics_a_tgnn['fpr'],
            'cross_ds_recall': metrics_b_tgnn['recall'],
            'cross_ds_f1': metrics_b_tgnn['f1'],
            'cross_ds_fpr': metrics_b_tgnn['fpr'],
            'cross_ds_recall_drop': float(recall_drop_tgnn)
        }
        
        # 2. Static GNN Baseline (Graph-aware, no temporal attention)
        static_model = StaticGNNBaseline(node_in_dim=6, hidden_dim=32)
        def eval_static(model, seqs):
            preds, targets = [], []
            for seq in seqs:
                y_hat = model.forward(seq['snapshots'][-1])
                preds.append(y_hat)
                targets.append(seq['target_y'])
            p_arr = np.concatenate(preds)
            t_arr = np.concatenate(targets)
            bin_p = (p_arr >= 0.5).astype(int)
            tp = np.sum((bin_p == 1) & (t_arr == 1))
            fn = np.sum((bin_p == 0) & (t_arr == 1))
            fp = np.sum((bin_p == 1) & (t_arr == 0))
            tn = np.sum((bin_p == 0) & (t_arr == 0))
            rec = tp / max(1, tp + fn)
            fpr = fp / max(1, fp + tn)
            prec = tp / max(1, tp + fp)
            f1 = 2 * prec * rec / max(1e-8, prec + rec)
            return rec, f1, fpr

        r_a_st, f1_a_st, fpr_a_st = eval_static(static_model, self.ds_a_test)
        r_b_st, f1_b_st, fpr_b_st = eval_static(static_model, self.ds_b_test)
        
        results['Static GNN Baseline'] = {
            'in_dist_recall': float(r_a_st),
            'in_dist_f1': float(f1_a_st),
            'in_dist_fpr': float(fpr_a_st),
            'cross_ds_recall': float(r_b_st),
            'cross_ds_f1': float(f1_b_st),
            'cross_ds_fpr': float(fpr_b_st),
            'cross_ds_recall_drop': float(r_a_st - r_b_st)
        }
        
        # 3. LSTM-Only Baseline (Temporal-aware, no graph structure)
        lstm_model = LSTMOnlyBaseline(node_in_dim=6, hidden_dim=32, history_len=5)
        def eval_lstm(model, seqs):
            preds, targets = [], []
            for seq in seqs:
                y_hat = model.forward(seq['snapshots'])
                preds.append(y_hat)
                targets.append(seq['target_y'])
            p_arr = np.concatenate(preds)
            t_arr = np.concatenate(targets)
            bin_p = (p_arr >= 0.5).astype(int)
            tp = np.sum((bin_p == 1) & (t_arr == 1))
            fn = np.sum((bin_p == 0) & (t_arr == 1))
            fp = np.sum((bin_p == 1) & (t_arr == 0))
            tn = np.sum((bin_p == 0) & (t_arr == 0))
            rec = tp / max(1, tp + fn)
            fpr = fp / max(1, fp + tn)
            prec = tp / max(1, tp + fp)
            f1 = 2 * prec * rec / max(1e-8, prec + rec)
            return rec, f1, fpr

        r_a_lstm, f1_a_lstm, fpr_a_lstm = eval_lstm(lstm_model, self.ds_a_test)
        r_b_lstm, f1_b_lstm, fpr_b_lstm = eval_lstm(lstm_model, self.ds_b_test)
        
        results['LSTM-Only Baseline'] = {
            'in_dist_recall': float(r_a_lstm),
            'in_dist_f1': float(f1_a_lstm),
            'in_dist_fpr': float(fpr_a_lstm),
            'cross_ds_recall': float(r_b_lstm),
            'cross_ds_f1': float(f1_b_lstm),
            'cross_ds_fpr': float(fpr_b_lstm),
            'cross_ds_recall_drop': float(r_a_lstm - r_b_lstm)
        }
        
        # 4. Tabular Random Forest Baseline
        rf = TabularRandomForestBaseline(n_estimators=30)
        X_train_rf = np.concatenate([seq['snapshots'][-1]['x'] for seq in self.ds_a_train])
        y_train_rf = np.concatenate([seq['target_y'] for seq in self.ds_a_train])
        rf.fit(X_train_rf, y_train_rf)
        
        def eval_rf(model, seqs):
            X_test = np.concatenate([seq['snapshots'][-1]['x'] for seq in seqs])
            y_test = np.concatenate([seq['target_y'] for seq in seqs])
            probs = model.predict_proba(X_test)
            bin_p = (probs >= 0.5).astype(int)
            tp = np.sum((bin_p == 1) & (y_test == 1))
            fn = np.sum((bin_p == 0) & (y_test == 1))
            fp = np.sum((bin_p == 1) & (y_test == 0))
            tn = np.sum((bin_p == 0) & (y_test == 0))
            rec = tp / max(1, tp + fn)
            fpr = fp / max(1, fp + tn)
            prec = tp / max(1, tp + fp)
            f1 = 2 * prec * rec / max(1e-8, prec + rec)
            return rec, f1, fpr

        r_a_rf, f1_a_rf, fpr_a_rf = eval_rf(rf, self.ds_a_test)
        r_b_rf, f1_b_rf, fpr_b_rf = eval_rf(rf, self.ds_b_test)
        
        results['Random Forest (Flow Features)'] = {
            'in_dist_recall': float(r_a_rf),
            'in_dist_f1': float(f1_a_rf),
            'in_dist_fpr': float(fpr_a_rf),
            'cross_ds_recall': float(r_b_rf),
            'cross_ds_f1': float(f1_b_rf),
            'cross_ds_fpr': float(fpr_b_rf),
            'cross_ds_recall_drop': float(r_a_rf - r_b_rf)
        }
        
        return results
