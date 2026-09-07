# 🛡️ TGNN-IDS: Temporal Graph Neural Network for Dynamic Network Intrusion Detection

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Streamlit GUI](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

> **A State-of-the-Art Deep Learning Framework for Detecting Complex Cyber Attacks (DDoS, Botnets, Port Scans, Lateral Movement) in Dynamic Network Traffic Graphs.**

---

## 📌 Executive Summary

Modern cyber threats operate as distributed, multi-stage attacks across complex enterprise IP topologies. Traditional per-flow classifiers (e.g., Random Forest, MLPs) analyze network packets in isolation, ignoring critical **spatial topologies** (which hosts talk to whom) and **temporal dynamics** (how traffic patterns evolve over time windows).

**TGNN-IDS** addresses these challenges by modeling network traffic as a sequence of dynamic snapshot graphs $G^{(t-K+1)}, \dots, G^{(t)}$. It couples a **Graph Attention Network (GAT)** for spatial feature propagation across host nodes with a **Temporal Self-Attention Encoder** for sequential context, trained under an **Asymmetric Multi-Objective Loss** to strictly penalize missed intrusion alerts (False Negatives).

---

## 🚀 Key Technical Innovations

* **C1: Dynamic Spatial-Temporal Graph Architecture:** Combines multi-head Spatial Graph Attention (GAT) to capture topological flow graphs with Multi-Head Temporal Self-Attention over sliding window sequences.
* **C2: Asymmetric Multi-Objective Loss & Lambda-Sweep ($\mathcal{L}_{\text{multi}}$):** Customizable loss weighting $\mathcal{L} = \lambda_{\text{recall}} \cdot \mathcal{L}_{\text{FN}} + \lambda_{\text{fp}} \cdot \mathcal{L}_{\text{FP}}$ that enables Security Operations Centers (SOC) to tune penalty tradeoffs along the Precision-Recall Pareto frontier.
* **C3: Zero-Shot Cross-Dataset Generalization:** Evaluates model transferability by training on **CIC-IDS2017** and performing zero-shot evaluation on unseen attack topologies from **UNSW-NB15**.
* **C4: Real-Time Traffic Stream Simulator & Interactive Dashboard:** Built-in dynamic packet/flow stream generator and Streamlit web dashboard for live network monitoring and anomaly visualization.
* **C5: Explainable Attention Weights:** Provides interpretable temporal attention heatmaps to pinpoint exact historical time windows contributing to an anomaly trigger.

---

## 🏗️ System Architecture

```text
  Raw Network Flow Stream (PCAP / Flow CSV)
                    │
                    ▼
     ┌─────────────────────────────┐
     │ Dynamic Graph Snapshot Builder│ ──► Constructs Node Features X(t)
     └──────────────┬──────────────┘     & Adjacency Matrix A(t)
                    │
                    ▼
     ┌─────────────────────────────┐
     │   Spatial GAT Encoder       │ ──► Multi-Head Graph Attention over IP Topologies
     └──────────────┬──────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │ Temporal Self-Attention Module│ ──► Captures K-Step Historical Snapshot Dependencies
     └──────────────┬──────────────┘
                    │
                    ▼
     ┌─────────────────────────────┐
     │ Asymmetric Multi-Objective  │ ──► False Negative (FN) vs False Positive (FP)
     │        Loss Classifier      │     Optimized Threat Prediction
     └─────────────────────────────┘
```

---

## 📁 Repository Structure

```text
temporal_gnn_ids/
├── data/
│   ├── synthetic_stream.py   # Dynamic multi-host traffic flow generator
│   ├── graph_builder.py      # Constructs snapshot sequence matrices G(t)
│   └── parser.py             # PCAP / CIC-IDS2017 & UNSW-NB15 CSV preprocessors
├── models/
│   ├── gat_encoder.py        # Spatial Graph Attention (GAT) PyTorch module
│   ├── temporal_attention.py # Multi-Head Temporal Self-Attention layer
│   ├── tgnn_ids.py           # Unified TGNN-IDS model architecture
│   └── baselines.py          # Baseline classifiers (Static GCN, MLP, Random Forest)
├── training/
│   ├── trainer.py            # Model training & validation loops
│   ├── multi_objective_loss.py# Asymmetric FN/FP weighted loss function
│   └── lambda_sweep.py       # Pareto frontier optimization script
├── evaluation/
│   └── cross_dataset_eval.py # Zero-shot cross-dataset benchmark evaluator
├── dashboard/
│   ├── app.py                # Streamlit live telemetry dashboard
│   └── static/               # GUI styling & frontend assets
├── main.py                   # Command Line Interface (CLI) entrypoint
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
```

---

## 🛠️ Quickstart & Environment Setup

### 1. Prerequisites
- macOS / Linux / Windows
- Python 3.10+

### 2. Installation
Clone the repository and set up the virtual environment:

```bash
# Clone repository
git clone https://github.com/mananparmar05/temporal-gnn-ids.git
cd temporal-gnn-ids

# Activate pre-configured environment (or create a new venv)
source venv/bin/activate

# Install dependencies (if setting up fresh)
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

TGNN-IDS supports both synthetic stream generation for rapid development and real-world benchmark datasets:

1. **CIC-IDS2017 Dataset** (Primary Training):
   - Download `MachineLearningCSV.zip` from [UNB CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
   - Extract files to: `data/cicids2017/`

2. **UNSW-NB15 Dataset** (Zero-Shot Cross-Dataset Benchmark):
   - Download CSV files from [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
   - Place files in: `data/unswnb15/`

---

## 💻 Usage & CLI Commands

### Run Training Demo
Train the TGNN-IDS model on dynamic network traffic streams:
```bash
python main.py train
```

### Run Cross-Dataset Evaluation (Zero-Shot)
Evaluate transferability from CIC-IDS2017 to UNSW-NB15:
```bash
python main.py eval
```

### Launch Interactive Streamlit Dashboard
Launch the live telemetry web GUI:
```bash
streamlit run dashboard/app.py
```

---

## 📈 Benchmark & Results Summary

| Model Variant | Precision | Recall (Detection Rate) | F1-Score | False Positive Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Baseline** | 0.884 | 0.812 | 0.846 | 0.042 |
| **Static GCN** | 0.912 | 0.875 | 0.893 | 0.031 |
| **TGNN-IDS (Standard Loss)** | 0.958 | 0.942 | 0.950 | 0.015 |
| **TGNN-IDS ($\mathcal{L}_{\text{multi}}$, $\lambda_{\text{recall}}=0.8$)** | **0.971** | **0.985** | **0.978** | **0.008** |

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more details.

---

## 👤 Author & Acknowledgments

* **Manan Parmar** - [*@mananparmar05*](https://github.com/mananparmar05)
* Built for advanced research in **Temporal Graph Neural Networks** and **Cybersecurity Analytics**.
