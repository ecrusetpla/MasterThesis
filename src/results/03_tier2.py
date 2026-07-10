# %%
# %%
# 03_tier2_threshold.py
# ===========================================================================
# Tier 2 — LSTM & GRU sequence models for monthly link-level classification.
#
# ADDITIONS VS PREVIOUS VERSION
# ──────────────────────────────
# 1. 10 fixed seeds — each model trained 10 times; results are aggregated.
# 2. 95% confidence intervals (normal approximation) across seeds.
# 3. Extended metrics: Precision, Recall, F1 (positive), Macro-F1,
#    PR-AUC, AUC-ROC, Balanced Accuracy, Confusion Matrix (TP/TN/FP/FN).
#
# Everything else (architecture, DataLoader, sequence builder, scaler,
# threshold selection, early stopping, training curves) is UNCHANGED
# from the working original.
# ===========================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score, precision_recall_curve,
)
import matplotlib.pyplot as plt

from feature_sets import FEATURE_SETS, TARGET, resolve_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = Path("../../data/processed/modelling_panel_THRESHOLD.parquet")
RESULTS_DIR = Path("../../results/tier2 THRESHOLD")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters  (unchanged from original)
# ---------------------------------------------------------------------------
SEQ_LEN     = 6
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 1e-3
PATIENCE    = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Fixed seeds
# ---------------------------------------------------------------------------
SEEDS = [42, 1, 7, 13, 21, 37, 55, 77, 99, 123]


# %%
# ---------------------------------------------------------------------------
# Load & sort
# ---------------------------------------------------------------------------
panel = pd.read_parquet(DATA_PATH).sort_values(["source", "target", "time_id"])


# %%
# ---------------------------------------------------------------------------
# Temporal split  70 / 10 / 20
# ---------------------------------------------------------------------------
n_times   = panel["time_id"].nunique()
train_cut = int(n_times * 0.70)
val_cut   = int(n_times * 0.80)

train_ids = sorted(panel["time_id"].unique())[:train_cut]
val_ids   = sorted(panel["time_id"].unique())[train_cut:val_cut]
test_ids  = sorted(panel["time_id"].unique())[val_cut:]

mask_train = panel["time_id"].isin(train_ids)


# %%
# ---------------------------------------------------------------------------
# Binary label — computed AFTER split, on training rows only
# ---------------------------------------------------------------------------
tau = panel.loc[mask_train, "proportion_disrupted"].median()
print(f"Disruption threshold tau (50th pct, train only): {tau:.4f}")

panel["Significant_Disruption"] = (
    panel["proportion_disrupted"] > tau
).astype(int)

print(
    f"Class balance:\n{panel['Significant_Disruption'].value_counts()}\n"
    f"Positive rate (train): {panel.loc[mask_train, 'Significant_Disruption'].mean():.3f}"
)

assert TARGET == "Significant_Disruption", (
    f"feature_sets.py has TARGET='{TARGET}' — update it to 'Significant_Disruption'."
)


# %%
# ---------------------------------------------------------------------------
# Dataset  (unchanged)
# ---------------------------------------------------------------------------
class LinkSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len):
        self.X, self.y = [], []
        for (src, tgt), grp in df.groupby(["source", "target"]):
            grp    = grp.sort_values("time_id")
            X_link = grp[feature_cols].values.astype(np.float32)
            y_link = grp[target_col].values.astype(np.float32)
            for i in range(seq_len, len(grp)):
                self.X.append(X_link[i - seq_len : i])
                self.y.append(y_link[i])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):          return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# %%
# ---------------------------------------------------------------------------
# Model architecture  (unchanged)
# ---------------------------------------------------------------------------
class DelayRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 rnn_type="LSTM", dropout=0.3):
        super().__init__()
        self.rnn_type = rnn_type
        RNNClass = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = RNNClass(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# %%
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimiser):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, targets = [], []
    for xb, yb in loader:
        p = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
        probs.extend(p)
        targets.extend(yb.numpy())
    return np.array(targets), np.array(probs)


def best_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thr[min(np.argmax(f1s), len(thr) - 1)])


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision":         precision_score(y_true, y_pred, zero_division=0),
        "Recall":            recall_score(y_true, y_pred, zero_division=0),
        "F1 (positive)":     f1_score(y_true, y_pred, zero_division=0),
        "Macro-F1":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "PR-AUC":            average_precision_score(y_true, y_prob),
        "AUC-ROC":           roc_auc_score(y_true, y_prob),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def ci95(values):
    arr  = np.array(values, dtype=float)
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    return float(arr.mean()), float(half)


# %%
# ---------------------------------------------------------------------------
# Metric columns to track across seeds
# ---------------------------------------------------------------------------
METRIC_COLS = [
    "Balanced Accuracy", "Precision", "Recall",
    "F1 (positive)", "Macro-F1", "PR-AUC", "AUC-ROC",
    "TP", "TN", "FP", "FN",
]

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_seed_rows = []   # one row per (model, feature_set, seed)
summary_rows  = []   # mean +/- CI per (model, feature_set)

for fs_name, fs_cols in FEATURE_SETS.items():
    feats = resolve_features(fs_cols, panel.columns)
    print(f"\n{'='*60}\nFeature set: {fs_name}  ({len(feats)} features)")

    # Scale — fit on training slice only
    scaler     = StandardScaler()
    train_data = panel[panel["time_id"].isin(train_ids)].copy()
    train_data[feats] = scaler.fit_transform(train_data[feats].fillna(0))

    val_data  = panel[panel["time_id"].isin(val_ids)].copy()
    val_data[feats]  = scaler.transform(val_data[feats].fillna(0))

    test_data = panel[panel["time_id"].isin(test_ids)].copy()
    test_data[feats] = scaler.transform(test_data[feats].fillna(0))

    # Datasets and loaders — built once per feature set (data is fixed)
    val_ds   = LinkSequenceDataset(val_data,   feats, TARGET, SEQ_LEN)
    test_ds  = LinkSequenceDataset(test_data,  feats, TARGET, SEQ_LEN)
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Class weight from training labels
    y_tr  = train_data[TARGET].astype(int)
    pos_w = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
        dtype=torch.float32,
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    for rnn_type in ["LSTM", "GRU"]:
        model_name   = rnn_type
        seed_metrics = {k: [] for k in METRIC_COLS}
        seed_thresholds = []
        # Training curves averaged across seeds (for the saved figure)
        all_val_ba_curves = []

        for seed in SEEDS:
            set_seed(seed)

            # Re-build train loader with this seed so shuffle order differs
            train_ds     = LinkSequenceDataset(train_data, feats, TARGET, SEQ_LEN)
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                worker_init_fn=lambda _: np.random.seed(seed),
            )

            model     = DelayRNN(len(feats), HIDDEN_SIZE, NUM_LAYERS, rnn_type).to(DEVICE)
            optimiser = optim.Adam(model.parameters(), lr=LR)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, patience=3, factor=0.5
            )

            best_val_ba, best_state, patience_ctr = 0.0, None, 0
            val_ba_history = []

            for epoch in range(1, EPOCHS + 1):
                tr_loss  = train_epoch(model, train_loader, criterion, optimiser)
                y_v, p_v = predict(model, val_loader)
                val_thr  = best_threshold(y_v, p_v)
                val_ba   = balanced_accuracy_score(y_v, (p_v >= val_thr).astype(int))
                scheduler.step(1 - val_ba)
                val_ba_history.append(val_ba)

                if val_ba > best_val_ba:
                    best_val_ba  = val_ba
                    best_thr     = val_thr
                    best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

            all_val_ba_curves.append(val_ba_history)

            # Evaluate on test set with best weights
            model.load_state_dict(best_state)
            y_te, p_te = predict(model, test_loader)
            y_pred     = (p_te >= best_thr).astype(int)
            m          = compute_metrics(y_te.astype(int), y_pred, p_te)

            for k in METRIC_COLS:
                seed_metrics[k].append(m[k])
            seed_thresholds.append(best_thr)

            all_seed_rows.append({
                "Model": model_name, "Feature Set": fs_name, "Seed": seed,
                "Threshold": best_thr,
                **{k: m[k] for k in METRIC_COLS},
            })

        # ------------------------------------------------------------------
        # Aggregate across seeds
        # ------------------------------------------------------------------
        row = {"Model": model_name, "Feature Set": fs_name}
        for k in METRIC_COLS:
            m_val, h_val = ci95(seed_metrics[k])
            row[f"{k} mean"] = round(m_val, 4)
            row[f"{k} CI95"] = round(h_val, 4)
        row["Threshold mean"] = round(float(np.mean(seed_thresholds)), 4)
        summary_rows.append(row)

        ba_m, ba_h   = ci95(seed_metrics["Balanced Accuracy"])
        f1_m, f1_h   = ci95(seed_metrics["F1 (positive)"])
        auc_m, auc_h = ci95(seed_metrics["AUC-ROC"])
        prauc_m, prauc_h = ci95(seed_metrics["PR-AUC"])
        print(
            f"\n  {rnn_type} | {fs_name}\n"
            f"    BA     = {ba_m:.4f} +/- {ba_h:.4f}\n"
            f"    F1+    = {f1_m:.4f} +/- {f1_h:.4f}\n"
            f"    AUC    = {auc_m:.4f} +/- {auc_h:.4f}\n"
            f"    PR-AUC = {prauc_m:.4f} +/- {prauc_h:.4f}"
        )

        # ------------------------------------------------------------------
        # Training curve (mean val BA across seeds)
        # ------------------------------------------------------------------
        max_len = max(len(c) for c in all_val_ba_curves)
        padded  = [
            c + [c[-1]] * (max_len - len(c)) for c in all_val_ba_curves
        ]
        mean_curve = np.mean(padded, axis=0)
        std_curve  = np.std(padded,  axis=0)

        fig, ax = plt.subplots(figsize=(7, 4))
        epochs_x = np.arange(1, max_len + 1)
        ax.plot(epochs_x, mean_curve, color="orange", label="Mean Val BA")
        ax.fill_between(
            epochs_x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.25, color="orange", label="+/- 1 SD",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"{rnn_type} Val BA across seeds — {fs_name}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(
            RESULTS_DIR / f"curve_{rnn_type}_{fs_name.replace('+', '')}.png",
            dpi=150,
        )
        plt.close()


# %%
# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df_seeds   = pd.DataFrame(all_seed_rows)
df_summary = pd.DataFrame(summary_rows)

df_seeds.to_csv(RESULTS_DIR / "tier2_all_seeds.csv",    index=False)
df_summary.to_csv(RESULTS_DIR / "tier2_summary_ci.csv", index=False)

print("\n" + "="*60)
print("TIER 2 — SUMMARY (mean +/- 95% CI across 10 seeds)")
print("="*60)
cols_show = [
    "Model", "Feature Set",
    "Balanced Accuracy mean", "Balanced Accuracy CI95",
    "F1 (positive) mean",     "F1 (positive) CI95",
    "Macro-F1 mean",          "Macro-F1 CI95",
    "AUC-ROC mean",           "AUC-ROC CI95",
    "PR-AUC mean",            "PR-AUC CI95",
    "Precision mean",         "Precision CI95",
    "Recall mean",            "Recall CI95",
]
print(df_summary[cols_show].to_string(index=False))
print(f"\nPer-seed detail -> {RESULTS_DIR / 'tier2_all_seeds.csv'}")
print(f"Summary         -> {RESULTS_DIR / 'tier2_summary_ci.csv'}")

# %%


# %%
# %%
# Confusion matrices -- Tier 2 (LSTM & GRU), averaged across seeds
# Reads from saved results; no need to re-run the training cell.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("../../results/tier2 THRESHOLD")

df = pd.read_csv(RESULTS_DIR / "tier2_all_seeds.csv")

MODEL_ORDER = ["LSTM", "GRU"]
FS_ORDER    = sorted(df["Feature Set"].unique())   # adjust order here if needed

n_models = len(MODEL_ORDER)
n_fs     = len(FS_ORDER)

fig, axes = plt.subplots(
    n_models, n_fs,
    figsize=(4.5 * n_fs, 4.0 * n_models),
    constrained_layout=True,
)

CMAP = "Blues"

for row_idx, model_name in enumerate(MODEL_ORDER):
    for col_idx, fs_name in enumerate(FS_ORDER):

        ax = axes[row_idx, col_idx]

        subset = df[(df["Model"] == model_name) & (df["Feature Set"] == fs_name)]

        tn = subset["TN"].mean()
        fp = subset["FP"].mean()
        fn = subset["FN"].mean()
        tp = subset["TP"].mean()

        cm_avg = np.array([[tn, fp],
                           [fn, tp]])

        # Row-normalise to percentages
        cm_pct = cm_avg / cm_avg.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap=CMAP, vmin=0, vmax=100)

        for i in range(2):
            for j in range(2):
                color = "white" if cm_pct[i, j] > 60 else "black"
                ax.text(j, i, f"{cm_pct[i, j]:.1f}%", ha="center", va="center",
                        fontsize=24, fontweight="bold", color=color)

        ax.set_xticks([])
        ax.set_yticks([])

        if row_idx == 0:
            ax.set_title(fs_name, fontsize=26, fontweight="bold", pad=8)

        if col_idx == 0:
            ax.set_ylabel(model_name, fontsize=26, fontweight="bold", labelpad=10)

# fig.suptitle(
#     "Average Confusion Matrices across 10 Seeds — Tier 2\n(cell values = row-normalised percentages; rows sum to 100 %)",
#     fontsize=13, fontweight="bold", y=1.02,
# )

out_path = RESULTS_DIR / "tier2_confusion_matrices.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
plt.show()
print(f"Saved -> {out_path}")

# %%
# %%
# Tier 2 -- Temporal Baseline Comparison
# ============================================================
# Baseline 1: Persistence  -- predict label(t) = label(t-1)
# Baseline 2: Rolling freq -- predict label(t) = mean(label[t-3:t]) > 0.5
#
# Both baselines are feature-agnostic, so they produce one set
# of numbers that applies across all feature sets.
# AUC-ROC / PR-AUC are reported for rolling freq only (it
# produces a probability). Persistence is hard-label only.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, average_precision_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = Path("../../data/processed/modelling_panel_THRESHOLD.parquet")
RESULTS_DIR = Path("../../results/tier2 THRESHOLD")

# ---------------------------------------------------------------------------
# Load panel and reproduce the same split + label as in training
# ---------------------------------------------------------------------------
panel = pd.read_parquet(DATA_PATH).sort_values(["source", "target", "time_id"])

all_times = sorted(panel["time_id"].unique())
n         = len(all_times)
train_ids = set(all_times[: int(n * 0.70)])
val_ids   = set(all_times[int(n * 0.70) : int(n * 0.80)])
test_ids  = set(all_times[int(n * 0.80) :])
mask_train = panel["time_id"].isin(train_ids)

tau = panel.loc[mask_train, "proportion_disrupted"].median()
panel["Significant_Disruption"] = (panel["proportion_disrupted"] > tau).astype(int)

# ---------------------------------------------------------------------------
# Build per-link label history (train + val only -- no leakage)
# ---------------------------------------------------------------------------
# We need label at t-1 and rolling 3-month freq up to t-1 for each test row.
# Sort each link's timeline and look back using only past observations.

TARGET = "Significant_Disruption"

panel_sorted = panel.sort_values(["source", "target", "time_id"]).copy()

# Shift label by 1 within each link to get "last month's label"
panel_sorted["label_lag1"] = (
    panel_sorted.groupby(["source", "target"])[TARGET].shift(1)
)

# Rolling 3-month mean of label (uses t-3, t-2, t-1 only via shift before rolling)
panel_sorted["label_roll3"] = (
    panel_sorted.groupby(["source", "target"])[TARGET]
    .shift(1)
    .rolling(3, min_periods=1)
    .mean()
    .values   # just extract the numpy array -- index is already aligned
)

# ---------------------------------------------------------------------------
# Restrict to test set rows that have valid lag (i.e. t-1 exists)
# ---------------------------------------------------------------------------
test_df = panel_sorted[panel_sorted["time_id"].isin(test_ids)].dropna(
    subset=["label_lag1"]
).copy()

y_true      = test_df[TARGET].values.astype(int)
y_pers      = test_df["label_lag1"].values.astype(int)   # persistence prediction
y_roll_prob = test_df["label_roll3"].values               # rolling freq as probability
y_roll_pred = (y_roll_prob >= 0.5).astype(int)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def metrics_hard(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "Balanced Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "Precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":            round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 (positive)":     round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Macro-F1":          round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "AUC-ROC":           round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else "N/A",
        "PR-AUC":            round(average_precision_score(y_true, y_prob), 4) if y_prob is not None else "N/A",
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    return out

# ---------------------------------------------------------------------------
# Compute baselines
# ---------------------------------------------------------------------------
res_pers = metrics_hard(y_true, y_pers,      y_prob=None)
res_roll = metrics_hard(y_true, y_roll_pred, y_prob=y_roll_prob)

# ---------------------------------------------------------------------------
# Load Tier 2 summary for comparison
# ---------------------------------------------------------------------------
df_summary = pd.read_csv(RESULTS_DIR / "tier2_summary_ci.csv")

SHOW_METRICS = [
    "Balanced Accuracy", "Precision", "Recall",
    "F1 (positive)", "Macro-F1", "AUC-ROC", "PR-AUC",
]

# Build a tidy comparison table
rows = []

# Baselines -- repeated across feature sets for easy side-by-side reading
for fs in sorted(df_summary["Feature Set"].unique()):
    rows.append({
        "Model": "Persistence (t-1 label)",
        "Feature Set": fs,
        **{m: res_pers[m] for m in SHOW_METRICS},
        "CI95": "N/A",
    })
    rows.append({
        "Model": "Rolling Freq 3m",
        "Feature Set": fs,
        **{m: res_roll[m] for m in SHOW_METRICS},
        "CI95": "N/A",
    })

# Tier 2 models
for _, r in df_summary.iterrows():
    rows.append({
        "Model": r["Model"],
        "Feature Set": r["Feature Set"],
        **{m: f"{r[f'{m} mean']} +/- {r[f'{m} CI95']}" for m in SHOW_METRICS},
        "CI95": "shown inline",
    })

df_compare = pd.DataFrame(rows).sort_values(["Feature Set", "Model"])

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
print("=" * 80)
print("TIER 2 vs TEMPORAL BASELINES")
print("Baselines are feature-agnostic (same number regardless of feature set)")
print("=" * 80)

print("\n--- BASELINES (raw numbers) ---")
print(f"{'Metric':<22} {'Persistence':>18} {'Rolling Freq 3m':>18}")
print("-" * 60)
for m in SHOW_METRICS:
    print(f"{m:<22} {str(res_pers[m]):>18} {str(res_roll[m]):>18}")

print("\n--- CONFUSION MATRICES ---")
print(f"Persistence:      TP={res_pers['TP']}  TN={res_pers['TN']}  FP={res_pers['FP']}  FN={res_pers['FN']}")
print(f"Rolling Freq 3m:  TP={res_roll['TP']}  TN={res_roll['TN']}  FP={res_roll['FP']}  FN={res_roll['FN']}")

print("\n--- TIER 2 MODELS (mean +/- 95% CI across 10 seeds) ---")
cols_show = ["Model", "Feature Set",
             "Balanced Accuracy mean", "Balanced Accuracy CI95",
             "F1 (positive) mean",     "F1 (positive) CI95",
             "AUC-ROC mean",           "AUC-ROC CI95",
             "PR-AUC mean",            "PR-AUC CI95"]
print(df_summary[cols_show].to_string(index=False))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
baseline_df = pd.DataFrame([
    {"Baseline": "Persistence (t-1 label)",  **{m: res_pers[m] for m in SHOW_METRICS},
     "TP": res_pers["TP"], "TN": res_pers["TN"], "FP": res_pers["FP"], "FN": res_pers["FN"]},
    {"Baseline": "Rolling Freq 3m",          **{m: res_roll[m] for m in SHOW_METRICS},
     "TP": res_roll["TP"], "TN": res_roll["TN"], "FP": res_roll["FP"], "FN": res_roll["FN"]},
])
baseline_df.to_csv(RESULTS_DIR / "tier2_temporal_baselines.csv", index=False)
print(f"\nBaseline results saved -> {RESULTS_DIR / 'tier2_temporal_baselines.csv'}")


