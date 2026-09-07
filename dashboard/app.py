"""
FastAPI Analyst Dashboard Backend
Serves live stream inference API, network topology graphs, 
temporal attention interpretability explanations, and evaluation benchmarks.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
import os

from data.synthetic_stream import TrafficStreamGenerator
from data.graph_builder import SnapshotGraphBuilder
from models.tgnn_ids import TGNN_IDS
from training.trainer import TGNNTrainer
from evaluation.cross_dataset_eval import CrossDatasetEvaluator
from training.lambda_sweep import run_lambda_sweep

app = FastAPI(title="TGNN-IDS Security Analyst Monitor")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

STATE = {
    'generator': None,
    'builder': None,
    'model': None,
    'sequences': [],
    'hosts': [],
    'current_step': 0,
    'benchmark_results': None,
    'pareto_results': None
}

@app.on_event("startup")
def init_system():
    print("Initializing TGNN-IDS Live Stream Monitor Engine...")
    gen = TrafficStreamGenerator(num_hosts=30, seed=42)
    windows, hosts = gen.generate_flow_stream(num_windows=25, window_duration_sec=30, attack_ratio=0.4)
    
    builder = SnapshotGraphBuilder(all_hosts=hosts, history_len=5)
    seqs, snapshots = builder.assemble_temporal_sequences(windows)
    
    model = TGNN_IDS(node_in_dim=6, hidden_dim=32, history_len=5)
    trainer = TGNNTrainer(model, lr=0.005, lambda_recall=0.7)
    
    train_seqs = seqs[:12]
    val_seqs = seqs[12:]
    for _ in range(10):
        trainer.train_epoch(train_seqs)
        
    STATE['generator'] = gen
    STATE['builder'] = builder
    STATE['model'] = model
    STATE['sequences'] = seqs
    STATE['hosts'] = hosts
    STATE['current_step'] = 0
    
    print("Calculating C3 Cross-Dataset benchmark & C2 Lambda-sweep metrics...")
    ds_b_gen = TrafficStreamGenerator(num_hosts=30, seed=99)
    ds_b_windows, ds_b_hosts = ds_b_gen.generate_flow_stream(num_windows=15, attack_ratio=0.5)
    ds_b_builder = SnapshotGraphBuilder(all_hosts=ds_b_hosts, history_len=5)
    ds_b_seqs, _ = ds_b_builder.assemble_temporal_sequences(ds_b_windows)
    
    evaluator = CrossDatasetEvaluator(
        dataset_a_train_seqs=train_seqs,
        dataset_a_test_seqs=val_seqs,
        dataset_b_zero_shot_seqs=ds_b_seqs
    )
    STATE['benchmark_results'] = evaluator.evaluate_all_models(epochs=8)
    STATE['pareto_results'] = run_lambda_sweep(train_seqs, val_seqs, epochs=8)
    print("TGNN-IDS Engine Initialization Complete!")

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    with open(os.path.join(static_dir, "index.html"), "r") as f:
        return f.read()

@app.get("/api/status")
def get_status():
    return {
        "status": "ONLINE",
        "num_hosts": len(STATE['hosts']),
        "total_time_steps": len(STATE['sequences']),
        "current_step": STATE['current_step'],
        "history_len": 5
    }

@app.get("/api/stream_step")
def get_stream_step(step: int = -1):
    if step < 0:
        step = STATE['current_step']
        STATE['current_step'] = (STATE['current_step'] + 1) % len(STATE['sequences'])
    else:
        step = step % len(STATE['sequences'])
        
    seq = STATE['sequences'][step]
    current_snap = seq['snapshots'][-1]
    
    y_hat_np, beta_np = STATE['model'].forward(seq['snapshots'])
    y_true_np = seq['target_y']
    
    nodes = []
    hosts = STATE['hosts']
    for idx, host in enumerate(hosts):
        score = float(y_hat_np[idx])
        is_attack = bool(y_true_np[idx] > 0.5)
        
        attn_weights = [float(w) for w in beta_np[idx]]
        
        nodes.append({
            "id": host,
            "label": host,
            "anomaly_score": round(score, 4),
            "is_anomaly": score >= 0.5,
            "ground_truth": "ATTACK" if is_attack else "BENIGN",
            "attention_weights": attn_weights
        })
        
    edges = []
    edge_index = current_snap['edge_index']
    if edge_index.shape[1] > 0:
        for e_i in range(edge_index.shape[1]):
            u = hosts[edge_index[0, e_i]]
            v = hosts[edge_index[1, e_i]]
            edges.append({"from": u, "to": v})
            
    alerts = [n for n in nodes if n["is_anomaly"]]
    
    return {
        "time_step": step,
        "nodes": nodes,
        "edges": edges,
        "alerts_count": len(alerts),
        "alerts": alerts
    }

@app.get("/api/lambda_sweep")
def get_lambda_sweep():
    return STATE['pareto_results']

@app.get("/api/benchmark")
def get_benchmark():
    return STATE['benchmark_results']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8000, reload=True)
