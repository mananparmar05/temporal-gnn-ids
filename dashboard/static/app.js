/* TGNN-IDS Analyst Dashboard Frontend Controller */

let network = null;
let nodesDataSet = null;
let edgesDataSet = null;
let isPlaying = false;
let playInterval = null;
let currentSelectedHost = null;

document.addEventListener("DOMContentLoaded", () => {
  initGraphCanvas();
  fetchStreamStep();
  fetchBenchmark();
  fetchLambdaSweep();

  document.getElementById("btn-next").addEventListener("click", () => {
    fetchStreamStep();
  });

  document.getElementById("btn-play").addEventListener("click", toggleAutoStream);
});

function initGraphCanvas() {
  const container = document.getElementById("graph-canvas");
  nodesDataSet = new vis.DataSet([]);
  edgesDataSet = new vis.DataSet([]);

  const data = {
    nodes: nodesDataSet,
    edges: edgesDataSet
  };

  const options = {
    nodes: {
      shape: "dot",
      size: 20,
      font: {
        color: "#f3f4f6",
        face: "JetBrains Mono",
        size: 11
      },
      borderWidth: 2
    },
    edges: {
      width: 1.5,
      color: { color: "rgba(255, 255, 255, 0.15)", highlight: "#3b82f6" },
      arrows: { to: { enabled: true, scaleFactor: 0.5 } },
      smooth: { type: "continuous" }
    },
    physics: {
      stabilization: false,
      barnesHut: {
        gravitationalConstant: -2000,
        centralGravity: 0.3,
        springLength: 95
      }
    },
    interaction: { hover: true }
  };

  network = new vis.Network(container, data, options);

  network.on("click", (params) => {
    if (params.nodes.length > 0) {
      const hostId = params.nodes[0];
      selectHostForAttention(hostId);
    }
  });
}

async function fetchStreamStep() {
  try {
    const res = await fetch("/api/stream_step");
    const data = await res.json();

    document.getElementById("time-step-tag").innerText = `Window: t=${data.time_step}`;
    document.getElementById("alerts-count-tag").innerText = `${data.alerts_count} Flagged`;

    // Update Vis.js Graph Nodes
    const updatedNodes = data.nodes.map((node) => {
      const isAnomaly = node.is_anomaly;
      const color = isAnomaly
        ? { background: "#ef4444", border: "#f87171", highlight: { background: "#dc2626", border: "#ffffff" } }
        : { background: "#1e40af", border: "#3b82f6", highlight: { background: "#2563eb", border: "#ffffff" } };

      return {
        id: node.id,
        label: `${node.label}\n(${node.anomaly_score.toFixed(2)})`,
        color: color,
        raw_data: node
      };
    });

    nodesDataSet.clear();
    nodesDataSet.add(updatedNodes);

    // Update Vis.js Graph Edges
    const updatedEdges = data.edges.map((e, idx) => ({
      id: `e_${idx}`,
      from: e.from,
      to: e.to
    }));

    edgesDataSet.clear();
    edgesDataSet.add(updatedEdges);

    // Update Anomaly Alert Sidebar List
    renderAlertList(data.alerts);

    // Update attention bar if host selected
    if (currentSelectedHost) {
      const hostNode = data.nodes.find((n) => n.id === currentSelectedHost);
      if (hostNode) {
        updateAttentionBars(hostNode);
      }
    } else if (data.alerts.length > 0) {
      selectHostForAttention(data.alerts[0].id, data.nodes);
    }

  } catch (err) {
    console.error("Error fetching stream step:", err);
  }
}

function renderAlertList(alerts) {
  const container = document.getElementById("alert-list");
  container.innerHTML = "";

  if (alerts.length === 0) {
    container.innerHTML = `
      <div style="font-size: 0.85rem; color: var(--text-muted); text-align: center; padding: 1.5rem 0;">
        No anomalies detected in current window.
      </div>`;
    return;
  }

  alerts.forEach((alert) => {
    const div = document.createElement("div");
    div.className = `alert-item ${currentSelectedHost === alert.id ? "selected" : ""}`;
    div.onclick = () => selectHostForAttention(alert.id);

    div.innerHTML = `
      <div class="alert-header">
        <span class="alert-host">${alert.id}</span>
        <span class="alert-score">Score: ${alert.anomaly_score}</span>
      </div>
      <div style="font-size: 0.75rem; color: var(--text-muted);">
        Ground Truth: <strong style="color: #fca5a5;">${alert.ground_truth}</strong>
      </div>
    `;
    container.appendChild(div);
  });
}

function selectHostForAttention(hostId, nodesList = null) {
  currentSelectedHost = hostId;
  let node = null;
  
  if (nodesList) {
    node = nodesList.find(n => n.id === hostId);
  } else {
    const visNode = nodesDataSet.get(hostId);
    if (visNode) node = visNode.raw_data;
  }

  if (node) {
    updateAttentionBars(node);
  }
}

function updateAttentionBars(node) {
  document.getElementById("attn-host-info").innerHTML = 
    `Host: <strong style="color: var(--accent-cyan);">${node.id}</strong> | Score: <strong>${node.anomaly_score}</strong>`;

  const container = document.getElementById("attn-bars-container");
  container.innerHTML = "";

  const weights = node.attention_weights || [0.1, 0.1, 0.1, 0.2, 0.5];
  const labels = ["t-4", "t-3", "t-2", "t-1", "t (Now)"];

  weights.forEach((w, idx) => {
    const col = document.createElement("div");
    col.className = "attn-bar-col";
    const heightPercent = Math.max(8, Math.round(w * 100));

    col.innerHTML = `
      <div style="font-size: 0.65rem; color: var(--accent-cyan); font-family: var(--font-mono);">${(w * 100).toFixed(0)}%</div>
      <div class="attn-bar-fill" style="height: ${heightPercent}%;"></div>
      <div class="attn-bar-label">${labels[idx]}</div>
    `;
    container.appendChild(col);
  });
}

async function fetchBenchmark() {
  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();
    const tbody = document.getElementById("benchmark-table-body");
    tbody.innerHTML = "";

    Object.keys(data).forEach((modelName) => {
      const row = data[modelName];
      const tr = document.createElement("tr");
      if (modelName.includes("TGNN-IDS")) tr.className = "highlight-row";

      const dropPct = (row.cross_ds_recall_drop * 100).toFixed(1);
      const dropBadge = row.cross_ds_recall_drop <= 0.05 
        ? `<span class="badge-pill badge-green">Low Drop (${dropPct}%)</span>`
        : `<span class="badge-pill badge-blue">-${dropPct}%</span>`;

      tr.innerHTML = `
        <td><strong>${modelName}</strong></td>
        <td>${(row.in_dist_recall * 100).toFixed(1)}%</td>
        <td>${(row.in_dist_f1 * 100).toFixed(1)}%</td>
        <td>${(row.cross_ds_recall * 100).toFixed(1)}%</td>
        <td>${(row.cross_ds_f1 * 100).toFixed(1)}%</td>
        <td>${dropBadge}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error fetching benchmark metrics:", err);
  }
}

async function fetchLambdaSweep() {
  try {
    const res = await fetch("/api/lambda_sweep");
    const data = await res.json();
    const tbody = document.getElementById("pareto-table-body");
    tbody.innerHTML = "";

    data.forEach((pt) => {
      const tr = document.createElement("tr");
      if (pt.lambda_recall === 0.7) tr.className = "highlight-row";

      tr.innerHTML = `
        <td><strong>${pt.lambda_recall.toFixed(1)}</strong></td>
        <td>${pt.lambda_fp.toFixed(1)}</td>
        <td>${(pt.recall * 100).toFixed(1)}%</td>
        <td>${(pt.fpr * 100).toFixed(2)}%</td>
        <td>${(pt.f1 * 100).toFixed(1)}%</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error fetching lambda sweep metrics:", err);
  }
}

function toggleAutoStream() {
  const btn = document.getElementById("btn-play");
  if (isPlaying) {
    clearInterval(playInterval);
    isPlaying = false;
    btn.classList.add("btn-secondary");
    btn.innerHTML = `<svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M11.596 8.697l-6.363 3.692c-.54.313-1.233-.066-1.233-.697V4.308c0-.63.692-1.01 1.23-.696l6.363 3.692a.802.802 0 0 1 0 1.393z"/></svg> Auto Stream`;
  } else {
    isPlaying = true;
    btn.classList.remove("btn-secondary");
    btn.innerHTML = `<svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M5.5 3.5A1.5 1.5 0 0 1 7 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5zm5 0A1.5 1.5 0 0 1 12 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5z"/></svg> Pause`;
    playInterval = setInterval(fetchStreamStep, 2500);
  }
}
