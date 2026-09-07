"""
TGNN-IDS All-in-One CLI Entrypoint & Runner
"""

import sys
import os
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_stream import TrafficStreamGenerator
from data.graph_builder import SnapshotGraphBuilder
from models.tgnn_ids import TGNN_IDS
from training.trainer import TGNNTrainer
from training.lambda_sweep import run_lambda_sweep
from evaluation.cross_dataset_eval import CrossDatasetEvaluator

def run_training_demo():
    print("=" * 70)
    print("  TGNN-IDS: Temporal Graph Neural Network Intrusion Detection")
    print("=" * 70)
    print("[1/3] Generating synthetic dynamic network traffic stream...")
    gen = TrafficStreamGenerator(num_hosts=35, seed=42)
    windows, hosts = gen.generate_flow_stream(num_windows=30, window_duration_sec=30, attack_ratio=0.35)
    print(f"      Generated {len(windows)} time windows over {len(hosts)} IP hosts.")
    
    print("\n[2/3] Assembling temporal snapshot sequences G(t-K+1)...G(t) [K=5]...")
    builder = SnapshotGraphBuilder(all_hosts=hosts, history_len=5)
    seqs, _ = builder.assemble_temporal_sequences(windows)
    
    train_seqs = seqs[:20]
    test_seqs = seqs[20:]
    
    print("\n[3/3] Training TGNN-IDS Model (GAT Spatial + Temporal Attention)...")
    model = TGNN_IDS(node_in_dim=6, hidden_dim=32, history_len=5)
    trainer = TGNNTrainer(model, lr=0.005, lambda_recall=0.7)
    
    for epoch in range(1, 11):
        loss, fn_loss, fp_loss = trainer.train_epoch(train_seqs)
        print(f"  Epoch {epoch:02d}/10 | Total Loss: {loss:.4f} | FN Loss (Missed): {fn_loss:.4f} | FP Loss (Alerts): {fp_loss:.4f}")
        
    print("\nEvaluating on held-out test windows...")
    metrics = trainer.evaluate(test_seqs)
    print("-" * 50)
    print(f"  Test Detection Recall : {metrics['recall']*100:.2f}%")
    print(f"  False Positive Rate   : {metrics['fpr']*100:.2f}%")
    print(f"  Test Precision        : {metrics['precision']*100:.2f}%")
    print(f"  Test F1 Score         : {metrics['f1']*100:.2f}%")
    print("-" * 50)
    
    if metrics['betas'] is not None:
        print("\nAttention Interpretability Spot-Check (Sample Host β_k Weights):")
        sample_beta = metrics['betas'][0]
        windows_labels = ["t-4", "t-3", "t-2", "t-1", "t (Now)"]
        for lbl, w in zip(windows_labels, sample_beta):
            print(f"   {lbl} : {'█' * int(w * 30)} ({w*100:.1f}%)")

def run_evaluation_demo():
    print("=" * 70)
    print("  Contribution C3: Cross-Dataset Zero-Shot Generalization Benchmark")
    print("=" * 70)
    gen_a = TrafficStreamGenerator(num_hosts=30, seed=42)
    win_a, hosts_a = gen_a.generate_flow_stream(num_windows=20, attack_ratio=0.4)
    builder_a = SnapshotGraphBuilder(all_hosts=hosts_a, history_len=5)
    seqs_a, _ = builder_a.assemble_temporal_sequences(win_a)
    
    gen_b = TrafficStreamGenerator(num_hosts=30, seed=99)
    win_b, hosts_b = gen_b.generate_flow_stream(num_windows=15, attack_ratio=0.5)
    builder_b = SnapshotGraphBuilder(all_hosts=hosts_b, history_len=5)
    seqs_b, _ = builder_b.assemble_temporal_sequences(win_b)
    
    evaluator = CrossDatasetEvaluator(
        dataset_a_train_seqs=seqs_a[:12],
        dataset_a_test_seqs=seqs_a[12:],
        dataset_b_zero_shot_seqs=seqs_b
    )
    
    results = evaluator.evaluate_all_models(epochs=10)
    print(f"\n{'Model Baseline':<32} | {'In-Dist Rec':<12} | {'Zero-Shot Rec':<13} | {'Recall Drop (C3)':<16}")
    print("-" * 80)
    for model_name, res in results.items():
        in_r = f"{res['in_dist_recall']*100:.1f}%"
        zs_r = f"{res['cross_ds_recall']*100:.1f}%"
        drop = f"{res['cross_ds_recall_drop']*100:.1f}%"
        print(f"{model_name:<32} | {in_r:<12} | {zs_r:<13} | {drop:<16}")

def run_dashboard_server():
    import uvicorn
    print("=" * 70)
    print("  Starting TGNN-IDS Security Analyst Monitor Dashboard...")
    print("  URL: http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TGNN-IDS Execution Suite")
    parser.add_argument("mode", choices=["train", "evaluate", "dashboard"], nargs="?", default="train")
    args = parser.parse_args()
    
    if args.mode == "train":
        run_training_demo()
    elif args.mode == "evaluate":
        run_evaluation_demo()
    elif args.mode == "dashboard":
        run_dashboard_server()
