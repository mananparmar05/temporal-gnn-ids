"""
TGNN-IDS All-in-One CLI Entrypoint & Runner
Supports training on synthetic traffic streams, real CIC-IDS2017, and UNSW-NB15 datasets.
"""

import sys
import os
import glob
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_stream import TrafficStreamGenerator
from data.graph_builder import SnapshotGraphBuilder
from data.parser import FlowLogParser
from models.tgnn_ids import TGNN_IDS
from training.trainer import TGNNTrainer
from training.lambda_sweep import run_lambda_sweep
from evaluation.cross_dataset_eval import CrossDatasetEvaluator


def run_training(dataset_name="cicids2017", epochs=10, history_len=5, lr=0.005, lambda_recall=0.7):
    print("=" * 75)
    print("  TGNN-IDS: Temporal Graph Neural Network Intrusion Detection")
    print("=" * 75)

    parser = FlowLogParser()

    if dataset_name.lower() in ["cicids2017", "cic"]:
        cic_files = sorted(glob.glob("data/cicids2017/*.csv"))
        if not cic_files:
            print("[Error] No CSV files found in data/cicids2017/")
            return

        target_file = cic_files[0]
        print(f"[1/4] Loading real CIC-IDS2017 dataset file: {os.path.basename(target_file)}...")
        df_raw = parser.parse_cicids2017_csv(target_file)

        # Slice traffic flows into discrete temporal snapshot windows
        chunk_size = max(500, len(df_raw) // 15)
        num_chunks = min(15, len(df_raw) // chunk_size)
        windows_df = [df_raw.iloc[i * chunk_size : (i + 1) * chunk_size].copy() for i in range(num_chunks)]

        hosts = list(set(df_raw['src_ip'].unique()) | set(df_raw['dst_ip'].unique()))
        print(f"      Parsed {len(df_raw)} network flows across {len(hosts)} IP hosts in {num_chunks} snapshot windows.")

    elif dataset_name.lower() in ["unswnb15", "unsw"]:
        unsw_files = sorted(glob.glob("data/unswnb15/*.csv"))
        if not unsw_files:
            print("[Error] No CSV files found in data/unswnb15/")
            return

        # Prefer training set if present
        train_files = [f for f in unsw_files if "training" in f.lower() or "testing" in f.lower()]
        target_file = train_files[0] if train_files else unsw_files[0]
        print(f"[1/4] Loading real UNSW-NB15 dataset file: {os.path.basename(target_file)}...")
        df_raw = parser.parse_unswnb15_csv(target_file)

        chunk_size = max(500, len(df_raw) // 15)
        num_chunks = min(15, len(df_raw) // chunk_size)
        windows_df = [df_raw.iloc[i * chunk_size : (i + 1) * chunk_size].copy() for i in range(num_chunks)]

        hosts = list(set(df_raw['src_ip'].unique()) | set(df_raw['dst_ip'].unique()))
        print(f"      Parsed {len(df_raw)} network flows across {len(hosts)} IP hosts in {num_chunks} snapshot windows.")

    else:
        print("[1/4] Generating synthetic dynamic network traffic stream...")
        gen = TrafficStreamGenerator(num_hosts=35, seed=42)
        window_dicts, hosts = gen.generate_flow_stream(num_windows=20, window_duration_sec=30, attack_ratio=0.35)
        windows_df = [w['df'] for w in window_dicts]
        print(f"      Generated {len(windows_df)} time windows over {len(hosts)} IP hosts.")

    print(f"\n[2/4] Assembling temporal snapshot sequences G(t-K+1)...G(t) [K={history_len}]...")
    builder = SnapshotGraphBuilder(all_hosts=hosts, history_len=history_len)
    window_data = [{'df': w} for w in windows_df]
    seqs, _ = builder.assemble_temporal_sequences(window_data)

    if len(seqs) < 2:
        print("[Error] Not enough snapshot sequences generated for training.")
        return

    split_idx = max(1, int(len(seqs) * 0.7))
    train_seqs = seqs[:split_idx]
    test_seqs = seqs[split_idx:]
    print(f"      Constructed {len(seqs)} total temporal sequences ({len(train_seqs)} train / {len(test_seqs)} test).")

    print("\n[3/4] Training TGNN-IDS Model (GAT Spatial + Temporal Attention)...")
    model = TGNN_IDS(node_in_dim=6, hidden_dim=32, history_len=history_len)
    trainer = TGNNTrainer(model, lr=lr, lambda_recall=lambda_recall)

    for epoch in range(1, epochs + 1):
        loss, fn_loss, fp_loss = trainer.train_epoch(train_seqs)
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Total Loss: {loss:.4f} | FN Loss (Missed): {fn_loss:.4f} | FP Loss (Alerts): {fp_loss:.4f}")

    print("\n[4/4] Evaluating on held-out test snapshot windows...")
    metrics = trainer.evaluate(test_seqs)
    print("-" * 55)
    print(f"  Test Detection Recall : {metrics['recall']*100:.2f}%")
    print(f"  False Positive Rate   : {metrics['fpr']*100:.2f}%")
    print(f"  Test Precision        : {metrics['precision']*100:.2f}%")
    print(f"  Test F1 Score         : {metrics['f1']*100:.2f}%")
    print("-" * 55)

    if metrics.get('betas') is not None and len(metrics['betas']) > 0:
        print("\nAttention Interpretability Spot-Check (Sample Host β_k Weights):")
        sample_beta = metrics['betas'][0]
        windows_labels = [f"t-{history_len - 1 - i}" if i < history_len - 1 else "t (Now)" for i in range(history_len)]
        for lbl, w in zip(windows_labels, sample_beta):
            print(f"   {lbl:<7} : {'█' * int(w * 30)} ({w * 100:.1f}%)")


def run_evaluation_demo():
    print("=" * 75)
    print("  Contribution C3: Cross-Dataset Zero-Shot Generalization Benchmark")
    print("=" * 75)
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
    print("=" * 75)
    print("  Starting TGNN-IDS Security Analyst Monitor Dashboard...")
    print("  URL: http://127.0.0.1:8000")
    print("=" * 75)
    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TGNN-IDS Execution Suite")
    parser.add_argument("mode", choices=["train", "evaluate", "dashboard"], nargs="?", default="train")
    parser.add_argument("--dataset", choices=["cicids2017", "unswnb15", "synthetic"], default="cicids2017",
                        help="Dataset to train on (default: cicids2017)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    if args.mode == "train":
        run_training(dataset_name=args.dataset, epochs=args.epochs)
    elif args.mode == "evaluate":
        run_evaluation_demo()
    elif args.mode == "dashboard":
        run_dashboard_server()
