# %%
# %%
# 04_tier3_threshold.py
# ===========================================================================
# Tier 3 -- GAT-only, GAT-LSTM, GAT-GRU graph models.
# Real models run with 10 fixed seeds; mean +/- 95% CI reported.
# Sanity checks (SC-A through SC-D) run with seed 42 only.
#
# FIXES APPLIED
# -------------
# 1. panel_raw saved before the loop so scaling always starts from clean data
# 2. cudnn.deterministic + cudnn.benchmark set inside set_seed
# 3. All other logic identical to the validated single-seed script
# ===========================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import random, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn    import GATv2Conv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score, precision_recall_curve
)

from feature_sets import FEATURE_SETS, TARGET, resolve_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = Path("../../data/processed/modelling_panel_THRESHOLD.parquet")
RESULTS_DIR = Path("../../results/tier3 THRESHOLD")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters & seeds
# ---------------------------------------------------------------------------
SEEDS       = [42, 1, 7, 13, 21, 37, 55, 77, 99, 123]
SANITY_SEED = 42
HIDDEN_DIM  = 32
GAT_HEADS   = 2
SEQ_LEN     = 6
EPOCHS      = 30
LR          = 5e-3
PATIENCE    = 5
LABEL_CUT   = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Load & sort
# ---------------------------------------------------------------------------
panel = pd.read_parquet(DATA_PATH).sort_values(["source", "target", "time_id"])

# ---------------------------------------------------------------------------
# Temporal split  70 / 10 / 20
# ---------------------------------------------------------------------------
all_times = sorted(panel["time_id"].unique())
n         = len(all_times)
train_ids = set(all_times[: int(n * 0.70)])
val_ids   = set(all_times[int(n * 0.70) : int(n * 0.80)])
test_ids  = set(all_times[int(n * 0.80) :])
mask_train = panel["time_id"].isin(train_ids)

print(f"Split sizes -- train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")

# ---------------------------------------------------------------------------
# Binary label
# ---------------------------------------------------------------------------
tau = panel.loc[mask_train, "proportion_disrupted"].median()
print(f"Disruption threshold tau (50th pct, train only): {tau:.4f}")

panel["Significant_Disruption"] = (
    panel["proportion_disrupted"] > tau
).astype(int)

print(f"Class balance:\n{panel[TARGET].value_counts()}")
print(f"Positive rate (train): {panel.loc[mask_train, TARGET].mean():.3f}")
assert TARGET == "Significant_Disruption"

# ---------------------------------------------------------------------------
# FIX 1: save a clean copy of panel before any scaling happens
# ---------------------------------------------------------------------------
panel_raw = panel.copy()

# ---------------------------------------------------------------------------
# Static graph
# ---------------------------------------------------------------------------
all_stations = sorted(set(panel["source"]) | set(panel["target"]))
node_idx     = {s: i for i, s in enumerate(all_stations)}
NUM_NODES    = len(all_stations)

edge_set = set()
for src, tgt in zip(panel["source"], panel["target"]):
    i, j = node_idx[src], node_idx[tgt]
    edge_set.add((i, j)); edge_set.add((j, i))
edge_index = torch.tensor(list(zip(*edge_set)), dtype=torch.long).to(DEVICE)

train_panel    = panel[panel["time_id"].isin(train_ids)]
train_edge_set = set()
for src, tgt in zip(train_panel["source"], train_panel["target"]):
    i, j = node_idx[src], node_idx[tgt]
    train_edge_set.add((i, j)); train_edge_set.add((j, i))
edge_index_train_only = torch.tensor(
    list(zip(*train_edge_set)), dtype=torch.long).to(DEVICE)

print(f"Full graph  -- Nodes: {NUM_NODES} | Edges: {edge_index.size(1)}")
print(f"Train-only  -- Edges: {edge_index_train_only.size(1)}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # FIX 2
    torch.backends.cudnn.benchmark     = False  # FIX 2

def binarise(y, cut=LABEL_CUT):
    return (np.asarray(y) >= cut).astype(int)

def best_threshold(y_true, y_prob, cut=LABEL_CUT):
    y_bin = binarise(y_true, cut)
    if len(np.unique(y_bin)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_bin, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thr[min(np.argmax(f1s), len(thr) - 1)])

def compute_metrics(y_true_bin, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred, labels=[0, 1]).ravel()
    return {
        "Balanced Accuracy": balanced_accuracy_score(y_true_bin, y_pred),
        "Precision":         precision_score(y_true_bin, y_pred, zero_division=0),
        "Recall":            recall_score(y_true_bin, y_pred, zero_division=0),
        "F1 (positive)":     f1_score(y_true_bin, y_pred, zero_division=0),
        "Macro-F1":          f1_score(y_true_bin, y_pred, average="macro", zero_division=0),
        "PR-AUC":            average_precision_score(y_true_bin, y_prob),
        "AUC-ROC":           roc_auc_score(y_true_bin, y_prob),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

def ci95(values):
    arr  = np.array(values, dtype=float)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    return arr.mean(), half

def make_pos_weight(ys_list, device):
    if not ys_list:
        return torch.tensor([1.0]).to(device)
    y_bin = binarise(torch.cat(ys_list).numpy())
    n_pos = max(int(y_bin.sum()), 1)
    n_neg = int((y_bin == 0).sum())
    return torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------
def build_snapshot(time_df, feature_cols):
    node_feat  = np.zeros((NUM_NODES, len(feature_cols)), dtype=np.float32)
    node_count = np.zeros(NUM_NODES,                      dtype=np.float32)
    node_label = np.zeros(NUM_NODES,                      dtype=np.float32)
    for _, row in time_df.iterrows():
        s_idx = node_idx.get(row["source"], -1)
        t_idx = node_idx.get(row["target"], -1)
        vals  = row[feature_cols].values.astype(np.float32)
        label = float(row[TARGET])
        if s_idx >= 0:
            node_feat[s_idx] += vals; node_count[s_idx] += 1; node_label[s_idx] += label
        if t_idx >= 0:
            node_feat[t_idx] += vals; node_count[t_idx] += 1; node_label[t_idx] += label
    node_count = np.maximum(node_count, 1)
    node_feat  /= node_count[:, None]
    node_label /= node_count
    return (torch.tensor(node_feat,  dtype=torch.float32),
            torch.tensor(node_label, dtype=torch.float32))

# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
class GATOnly(nn.Module):
    def __init__(self, in_channels, hidden, heads=GAT_HEADS):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels,    hidden, heads=heads, concat=True)
        self.gat2 = GATv2Conv(hidden * heads, hidden, heads=1,     concat=False)
        self.head = nn.Linear(hidden, 1)
        self.act  = nn.ELU()
    def forward(self, x, ei):
        x = self.act(self.gat1(x, ei))
        x = self.act(self.gat2(x, ei))
        return self.head(x).squeeze(-1)

class GATRNNCell(nn.Module):
    def __init__(self, in_channels, hidden, heads=GAT_HEADS, rnn_type="LSTM"):
        super().__init__()
        self.gat      = GATv2Conv(in_channels, hidden, heads=heads, concat=False)
        self.act      = nn.ELU()
        RNNCell       = nn.LSTMCell if rnn_type == "LSTM" else nn.GRUCell
        self.rnn      = RNNCell(hidden, hidden)
        self.rnn_type = rnn_type
    def forward(self, x, ei, h, c=None):
        spatial = self.act(self.gat(x, ei))
        if self.rnn_type == "LSTM":
            h, c = self.rnn(spatial, (h, c)); return h, c
        h = self.rnn(spatial, h); return h, None

class GATRNNModel(nn.Module):
    def __init__(self, in_channels, hidden, heads=GAT_HEADS, rnn_type="LSTM"):
        super().__init__()
        self.cell     = GATRNNCell(in_channels, hidden, heads, rnn_type)
        self.head     = nn.Linear(hidden, 1)
        self.hidden   = hidden
        self.rnn_type = rnn_type
    def forward(self, x_sequence, ei):
        n = x_sequence[0].size(0)
        h = torch.zeros(n, self.hidden, device=DEVICE)
        c = torch.zeros(n, self.hidden, device=DEVICE) if self.rnn_type == "LSTM" else None
        for x_t in x_sequence:
            h, c = self.cell(x_t.to(DEVICE), ei, h, c)
        return self.head(h).squeeze(-1)

# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------
def train_epoch_gatonly(model, xs, ys, criterion, opt, ei):
    model.train(); total = 0.0
    for i in range(len(xs) - 1):
        opt.zero_grad()
        loss = criterion(model(xs[i].to(DEVICE), ei), ys[i + 1].to(DEVICE))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item()
    return total / max(len(xs) - 1, 1)

def train_epoch_gatrnn(model, seqs, targets, criterion, opt, ei):
    model.train(); total = 0.0
    for x_seq, y_true in zip(seqs, targets):
        opt.zero_grad()
        loss = criterion(model([x.to(DEVICE) for x in x_seq], ei), y_true.to(DEVICE))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item()
    return total / max(len(seqs), 1)

@torch.no_grad()
def eval_gatonly(model, xs, ys, ei):
    model.eval()
    probs, labels = [], []
    for i in range(len(xs) - 1):
        probs.extend(torch.sigmoid(model(xs[i].to(DEVICE), ei)).cpu().numpy())
        labels.extend(ys[i + 1].numpy())
    return np.array(labels), np.array(probs)

@torch.no_grad()
def eval_gatrnn(model, seqs, targets, ei):
    model.eval()
    probs, labels = [], []
    for x_seq, y_true in zip(seqs, targets):
        probs.extend(torch.sigmoid(
            model([x.to(DEVICE) for x in x_seq], ei)).cpu().numpy())
        labels.extend(y_true.numpy())
    return np.array(labels), np.array(probs)

# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------
def train_and_evaluate(
    arch, feats, seed,
    train_seqs, train_seq_y, val_seqs, val_seq_y, test_seqs, test_seq_y,
    train_xs_gat, train_ys_gat, val_xs_gat, val_ys_gat, test_xs_gat, test_ys_gat,
    criterion_rnn, criterion_gat, ei, label="",
):
    set_seed(seed)
    if arch == "GAT-only":
        model     = GATOnly(len(feats), HIDDEN_DIM).to(DEVICE)
        criterion = criterion_gat
    elif arch == "GAT-LSTM":
        model     = GATRNNModel(len(feats), HIDDEN_DIM, rnn_type="LSTM").to(DEVICE)
        criterion = criterion_rnn
    else:
        model     = GATRNNModel(len(feats), HIDDEN_DIM, rnn_type="GRU").to(DEVICE)
        criterion = criterion_rnn

    opt       = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=3, factor=0.5)
    best_ba, best_thr, best_state, pat = 0.0, 0.5, None, 0

    for _ in range(EPOCHS):
        if arch == "GAT-only":
            train_epoch_gatonly(model, train_xs_gat, train_ys_gat, criterion, opt, ei)
            y_v, p_v = eval_gatonly(model, val_xs_gat, val_ys_gat, ei)
        else:
            train_epoch_gatrnn(model, train_seqs, train_seq_y, criterion, opt, ei)
            y_v, p_v = eval_gatrnn(model, val_seqs, val_seq_y, ei)

        thr = best_threshold(y_v, p_v)
        ba  = balanced_accuracy_score(binarise(y_v), (p_v >= thr).astype(int))
        scheduler.step(ba)
        if ba > best_ba:
            best_ba, best_thr, pat = ba, thr, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= PATIENCE:
            break

    model.load_state_dict(best_state)
    if arch == "GAT-only":
        y_te, p_te = eval_gatonly(model, test_xs_gat, test_ys_gat, ei)
    else:
        y_te, p_te = eval_gatrnn(model, test_seqs, test_seq_y, ei)

    y_te_bin = binarise(y_te)
    y_pred   = (p_te >= best_thr).astype(int)
    m        = compute_metrics(y_te_bin, y_pred, p_te)
    m["label"]     = label
    m["Threshold"] = best_thr
    return m

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
METRIC_COLS = [
    "Balanced Accuracy", "Precision", "Recall",
    "F1 (positive)", "Macro-F1", "PR-AUC", "AUC-ROC",
    "TP", "TN", "FP", "FN",
]

all_seed_rows = []
summary_rows  = []
sanity_rows   = []

for fs_name, fs_cols in FEATURE_SETS.items():
    feats = resolve_features(fs_cols, panel.columns)
    print(f"\n{'='*60}")
    print(f"Feature set: {fs_name}  ({len(feats)} features)")

    # FIX 1: always scale from the clean panel_raw, never from a
    # previously scaled version
    scaler       = StandardScaler()
    panel_scaled = panel_raw.copy()
    panel_scaled.loc[mask_train,  feats] = scaler.fit_transform(
        panel_raw.loc[mask_train,  feats].fillna(0))
    panel_scaled.loc[~mask_train, feats] = scaler.transform(
        panel_raw.loc[~mask_train, feats].fillna(0))

    all_xs, all_ys, vtids = [], [], []
    for t in all_times:
        tdf = panel_scaled[panel_scaled["time_id"] == t]
        if len(tdf) == 0:
            continue
        x, y = build_snapshot(tdf, feats)
        all_xs.append(x); all_ys.append(y); vtids.append(t)

    train_seqs, train_seq_y = [], []
    val_seqs,   val_seq_y   = [], []
    test_seqs,  test_seq_y  = [], []
    for i in range(len(all_xs) - SEQ_LEN):
        tid   = vtids[i + SEQ_LEN]
        x_seq = all_xs[i : i + SEQ_LEN]
        y_t   = all_ys[i + SEQ_LEN]
        if tid in train_ids:   train_seqs.append(x_seq); train_seq_y.append(y_t)
        elif tid in val_ids:   val_seqs.append(x_seq);   val_seq_y.append(y_t)
        elif tid in test_ids:  test_seqs.append(x_seq);  test_seq_y.append(y_t)

    train_xs_gat, train_ys_gat = [], []
    val_xs_gat,   val_ys_gat   = [], []
    test_xs_gat,  test_ys_gat  = [], []
    for i in range(len(all_xs) - 1):
        tid = vtids[i + 1]
        if tid in train_ids:
            train_xs_gat.append(all_xs[i]); train_ys_gat.append(all_ys[i + 1])
        elif tid in val_ids:
            val_xs_gat.append(all_xs[i]);   val_ys_gat.append(all_ys[i + 1])
        elif tid in test_ids:
            test_xs_gat.append(all_xs[i]);  test_ys_gat.append(all_ys[i + 1])

    pw_rnn        = make_pos_weight(train_seq_y,  DEVICE)
    pw_gat        = make_pos_weight(train_ys_gat, DEVICE)
    criterion_rnn = nn.BCEWithLogitsLoss(pos_weight=pw_rnn)
    criterion_gat = nn.BCEWithLogitsLoss(pos_weight=pw_gat)

    # ==================================================================
    # REAL MODEL RUNS -- 10 seeds
    # ==================================================================
    for arch in ["GAT-only", "GAT-LSTM", "GAT-GRU"]:
        seed_metrics = {k: [] for k in METRIC_COLS}
        for seed in SEEDS:
            m = train_and_evaluate(
                arch, feats, seed,
                train_seqs, train_seq_y, val_seqs, val_seq_y, test_seqs, test_seq_y,
                train_xs_gat, train_ys_gat, val_xs_gat, val_ys_gat,
                test_xs_gat,  test_ys_gat,
                criterion_rnn, criterion_gat, ei=edge_index, label="REAL",
            )
            for k in METRIC_COLS:
                seed_metrics[k].append(m[k])
            all_seed_rows.append({
                "Model": arch, "Feature Set": fs_name, "Seed": seed,
                **{k: m[k] for k in METRIC_COLS},
            })

        row = {"Model": arch, "Feature Set": fs_name}
        for k in METRIC_COLS:
            mv, hv = ci95(seed_metrics[k])
            row[f"{k} mean"] = round(mv, 4)
            row[f"{k} CI95"] = round(hv, 4)
        summary_rows.append(row)

        ba_m, ba_h   = ci95(seed_metrics["Balanced Accuracy"])
        auc_m, auc_h = ci95(seed_metrics["AUC-ROC"])
        print(f"  {arch}  BA={ba_m:.4f}+-{ba_h:.4f}  AUC={auc_m:.4f}+-{auc_h:.4f}")

    # ==================================================================
    # SANITY CHECKS -- seed 42 only
    # ==================================================================
    rng_a = np.random.default_rng(SANITY_SEED)

    def shuffle_labels_list(ys, rng):
        return [y[rng.permutation(len(y))] for y in ys]

    # SC-A
    tr_sy_a  = shuffle_labels_list(train_seq_y,  rng_a)
    val_sy_a = shuffle_labels_list(val_seq_y,    rng_a)
    te_sy_a  = shuffle_labels_list(test_seq_y,   rng_a)
    tr_yg_a  = shuffle_labels_list(train_ys_gat, rng_a)
    val_yg_a = shuffle_labels_list(val_ys_gat,   rng_a)
    te_yg_a  = shuffle_labels_list(test_ys_gat,  rng_a)
    pw_rnn_a = make_pos_weight(tr_sy_a,  DEVICE)
    pw_gat_a = make_pos_weight(tr_yg_a,  DEVICE)

    for arch in ["GAT-only", "GAT-LSTM", "GAT-GRU"]:
        m = train_and_evaluate(
            arch, feats, SANITY_SEED,
            train_seqs, tr_sy_a, val_seqs, val_sy_a, test_seqs, te_sy_a,
            train_xs_gat, tr_yg_a, val_xs_gat, val_yg_a, test_xs_gat, te_yg_a,
            nn.BCEWithLogitsLoss(pos_weight=pw_rnn_a),
            nn.BCEWithLogitsLoss(pos_weight=pw_gat_a),
            ei=edge_index, label="SC-A: shuffled labels",
        )
        sanity_rows.append({"Model": arch, "Feature Set": fs_name, **m})

    # SC-B
    rng_b = np.random.default_rng(SANITY_SEED)
    perm  = rng_b.permutation(len(all_xs))
    xs_b  = [all_xs[p] for p in perm]
    ys_b  = [all_ys[p] for p in perm]

    tr_s_b, tr_sy_b     = [], []
    val_s_b, val_sy_b   = [], []
    te_s_b,  te_sy_b    = [], []
    tr_xg_b, tr_yg_b    = [], []
    val_xg_b, val_yg_b  = [], []
    te_xg_b,  te_yg_b   = [], []

    for i in range(len(xs_b) - SEQ_LEN):
        tid = vtids[i + SEQ_LEN]
        xs  = xs_b[i : i + SEQ_LEN]; yt = ys_b[i + SEQ_LEN]
        if tid in train_ids:   tr_s_b.append(xs);   tr_sy_b.append(yt)
        elif tid in val_ids:   val_s_b.append(xs);  val_sy_b.append(yt)
        elif tid in test_ids:  te_s_b.append(xs);   te_sy_b.append(yt)
    for i in range(len(xs_b) - 1):
        tid = vtids[i + 1]
        if tid in train_ids:   tr_xg_b.append(xs_b[i]);  tr_yg_b.append(ys_b[i+1])
        elif tid in val_ids:   val_xg_b.append(xs_b[i]); val_yg_b.append(ys_b[i+1])
        elif tid in test_ids:  te_xg_b.append(xs_b[i]);  te_yg_b.append(ys_b[i+1])

    for arch in ["GAT-only", "GAT-LSTM", "GAT-GRU"]:
        m = train_and_evaluate(
            arch, feats, SANITY_SEED,
            tr_s_b, tr_sy_b, val_s_b, val_sy_b, te_s_b, te_sy_b,
            tr_xg_b, tr_yg_b, val_xg_b, val_yg_b, te_xg_b, te_yg_b,
            criterion_rnn, criterion_gat,
            ei=edge_index, label="SC-B: shuffled months",
        )
        sanity_rows.append({"Model": arch, "Feature Set": fs_name, **m})

    # SC-C
    topo_cols = [
        "topo_src_degree", "topo_tgt_degree",
        "topo_src_betweenness", "topo_tgt_betweenness",
        "topo_src_closeness", "topo_tgt_closeness",
        "topo_src_clustering", "topo_tgt_clustering",
        "topo_src_eigenvector", "topo_tgt_eigenvector",
        "topo_edge_betweenness", "topo_common_neighbours",
    ]
    topo_feats = resolve_features(topo_cols, panel.columns)
    if topo_feats:
        scaler_c = StandardScaler()
        panel_c  = panel_raw.copy()   # FIX 1: from panel_raw
        panel_c.loc[mask_train,  topo_feats] = scaler_c.fit_transform(
            panel_raw.loc[mask_train,  topo_feats].fillna(0))
        panel_c.loc[~mask_train, topo_feats] = scaler_c.transform(
            panel_raw.loc[~mask_train, topo_feats].fillna(0))
        xs_c, ys_c, vt_c = [], [], []
        for t in all_times:
            tdf = panel_c[panel_c["time_id"] == t]
            if len(tdf) == 0: continue
            x, y = build_snapshot(tdf, topo_feats)
            xs_c.append(x); ys_c.append(y); vt_c.append(t)
        tr_xc, tr_yc, val_xc, val_yc, te_xc, te_yc = [], [], [], [], [], []
        for i in range(len(xs_c) - 1):
            tid = vt_c[i + 1]
            if tid in train_ids:   tr_xc.append(xs_c[i]);  tr_yc.append(ys_c[i+1])
            elif tid in val_ids:   val_xc.append(xs_c[i]); val_yc.append(ys_c[i+1])
            elif tid in test_ids:  te_xc.append(xs_c[i]);  te_yc.append(ys_c[i+1])
        pw_c   = make_pos_weight(tr_yc, DEVICE)
        crit_c = nn.BCEWithLogitsLoss(pos_weight=pw_c)
        m = train_and_evaluate(
            "GAT-only", topo_feats, SANITY_SEED,
            [], [], [], [], [], [],
            tr_xc, tr_yc, val_xc, val_yc, te_xc, te_yc,
            crit_c, crit_c, ei=edge_index_train_only,
            label="SC-C: topo-only + train graph",
        )
        sanity_rows.append({"Model": "GAT-only", "Feature Set": fs_name, **m})

    # SC-D
    bl_cols  = ["topo_src_degree", "topo_tgt_degree", "rides_planned"]
    bl_feats = resolve_features(bl_cols, panel.columns)
    if bl_feats:
        scaler_d = StandardScaler()
        panel_d  = panel_raw.copy()   # FIX 1: from panel_raw
        panel_d.loc[mask_train,  bl_feats] = scaler_d.fit_transform(
            panel_raw.loc[mask_train,  bl_feats].fillna(0))
        panel_d.loc[~mask_train, bl_feats] = scaler_d.transform(
            panel_raw.loc[~mask_train, bl_feats].fillna(0))
        xs_d, ys_d, vt_d = [], [], []
        for t in all_times:
            tdf = panel_d[panel_d["time_id"] == t]
            if len(tdf) == 0: continue
            x, y = build_snapshot(tdf, bl_feats)
            xs_d.append(x); ys_d.append(y); vt_d.append(t)
        Xtr_d, ytr_d   = [], []
        Xval_d, yval_d = [], []
        Xte_d, yte_d   = [], []
        for i, t in enumerate(vt_d):
            xn = xs_d[i].numpy(); yn = binarise(ys_d[i].numpy())
            if t in train_ids:   Xtr_d.append(xn);  ytr_d.append(yn)
            elif t in val_ids:   Xval_d.append(xn);  yval_d.append(yn)
            elif t in test_ids:  Xte_d.append(xn);   yte_d.append(yn)
        Xtr_d  = np.vstack(Xtr_d);  ytr_d  = np.concatenate(ytr_d)
        Xval_d = np.vstack(Xval_d); yval_d = np.concatenate(yval_d)
        Xte_d  = np.vstack(Xte_d);  yte_d  = np.concatenate(yte_d)
        lr_d   = LogisticRegression(class_weight="balanced", max_iter=1000,
                                    random_state=SANITY_SEED)
        lr_d.fit(Xtr_d, ytr_d)
        vp_d                 = lr_d.predict_proba(Xval_d)[:, 1]
        prec_d, rec_d, thr_d = precision_recall_curve(yval_d, vp_d)
        f1s_d                = 2 * prec_d * rec_d / (prec_d + rec_d + 1e-8)
        opt_d                = float(thr_d[min(np.argmax(f1s_d), len(thr_d) - 1)])
        tp_d                 = lr_d.predict_proba(Xte_d)[:, 1]
        ypred_d              = (tp_d >= opt_d).astype(int)
        m_d                  = compute_metrics(yte_d, ypred_d, tp_d)
        m_d["label"]         = "SC-D: degree/service baseline"
        m_d["Threshold"]     = opt_d
        sanity_rows.append({"Model": "LogReg-baseline", "Feature Set": fs_name, **m_d})

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df_seeds   = pd.DataFrame(all_seed_rows)
df_summary = pd.DataFrame(summary_rows)
df_sanity  = pd.DataFrame(sanity_rows)

df_seeds.to_csv(RESULTS_DIR / "tier3_real_all_seeds.csv",    index=False)
df_summary.to_csv(RESULTS_DIR / "tier3_real_summary_ci.csv", index=False)
df_sanity.to_csv(RESULTS_DIR / "tier3_sanity_seed42.csv",    index=False)

print("\n" + "="*60)
print("TIER 3 REAL -- SUMMARY (mean +/- 95% CI, 10 seeds)")
cols_show = ["Model", "Feature Set",
             "Balanced Accuracy mean", "Balanced Accuracy CI95",
             "F1 (positive) mean",     "F1 (positive) CI95",
             "AUC-ROC mean",           "AUC-ROC CI95",
             "PR-AUC mean",            "PR-AUC CI95"]
print(df_summary[cols_show].to_string(index=False))

print("\n" + "="*60)
print("TIER 3 SANITY CHECKS (seed 42 only)")
print(df_sanity[["Model", "Feature Set", "label",
                  "Balanced Accuracy", "F1 (positive)", "AUC-ROC"]
                ].round(4).to_string(index=False))
print(f"\nAll results saved to {RESULTS_DIR}")


