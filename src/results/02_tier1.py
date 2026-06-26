# %%
# %%
# 02_tier1_threshold.py
# ===========================================================================
# Tier 1 -- link-level classifiers on the monthly panel.
# Runs each model with 10 fixed seeds and reports mean +/- 95% CI.
#
# CROSS-VALIDATION NOTE
# ---------------------
# Gradient Boosting uses walk-forward time-series cross-validation
# (expanding window, 5 folds) on the combined train+val period.
# Each fold's validation window is strictly after its training window,
# preserving temporal order. The reported metrics are averaged across
# folds AND seeds (10 seeds x 5 folds = 50 evaluations).
# Logistic Regression and Random Forest use the fixed train/val/test
# split as before.
# ===========================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import random, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score, precision_recall_curve,
)

from feature_sets import FEATURE_SETS, TARGET, resolve_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = Path("C:/Users/EduardCP/Documents/GitHub/MasterThesis/data/processed/modelling_panel_THRESHOLD.parquet")
RESULTS_DIR = Path("C:/Users/EduardCP/Documents/GitHub/MasterThesis/results/tier1 THRESHOLD")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility -- 10 fixed seeds
# ---------------------------------------------------------------------------
SEEDS = [42, 1, 7, 13, 21, 37, 55, 77, 99, 123]

# ---------------------------------------------------------------------------
# Walk-forward CV folds (Gradient Boosting only)
# ---------------------------------------------------------------------------
N_CV_FOLDS = 5      # number of temporal folds within train+val period

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
mask_val   = panel["time_id"].isin(val_ids)
mask_test  = panel["time_id"].isin(test_ids)

# ---------------------------------------------------------------------------
# Binary label -- fit threshold on training rows only
# ---------------------------------------------------------------------------
tau = panel.loc[mask_train, "proportion_disrupted"].median()
print(f"Disruption threshold tau (50th pct, train only): {tau:.4f}")

panel["Significant_Disruption"] = (
    panel["proportion_disrupted"] > tau
).astype(int)

print(f"Class balance (full):\n{panel[TARGET].value_counts()}")
print(f"Positive rate (train): {panel.loc[mask_train, TARGET].mean():.3f}")

assert TARGET == "Significant_Disruption"

# ---------------------------------------------------------------------------
# Walk-forward fold builder
# Operates on the train+val time IDs only; test is never touched.
# Returns list of (train_time_ids, val_time_ids) tuples.
# ---------------------------------------------------------------------------
def make_walkforward_folds(time_ids_sorted, n_folds):
    """
    Expanding-window walk-forward splits.
    time_ids_sorted : sorted list of time_id values (train+val period only).
    n_folds         : number of folds.

    Fold k uses:
      train -> time_ids[0 : min_train + k * step]
      val   -> time_ids[min_train + k * step : min_train + (k+1) * step]

    min_train is set to 60% of the available time_ids so every fold
    has a meaningful training window.
    """
    T         = len(time_ids_sorted)
    min_train = int(T * 0.60)
    remaining = T - min_train
    step      = max(remaining // n_folds, 1)

    folds = []
    for k in range(n_folds):
        tr_end  = min_train + k * step
        val_end = min(tr_end + step, T)
        if tr_end >= T:
            break
        tr_ids  = set(time_ids_sorted[:tr_end])
        val_ids_fold = set(time_ids_sorted[tr_end:val_end])
        if val_ids_fold:
            folds.append((tr_ids, val_ids_fold))
    return folds

# time IDs available for cross-validation (train + val, not test)
cv_time_ids = sorted(train_ids | val_ids)
cv_folds    = make_walkforward_folds(cv_time_ids, N_CV_FOLDS)
print(f"\nWalk-forward CV: {len(cv_folds)} folds over {len(cv_time_ids)} time periods")
for i, (tr, vl) in enumerate(cv_folds):
    print(f"  Fold {i+1}: train={len(tr)} periods, val={len(vl)} periods")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

def best_threshold_f1(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thr[min(np.argmax(f1s), len(thr) - 1)])

def ci95(values):
    arr  = np.array(values, dtype=float)
    m    = arr.mean()
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
    return m, half

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = {
    "Logistic Regression": lambda seed: LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=seed),
    "Random Forest": lambda seed: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
    "Gradient Boosting": lambda seed: GradientBoostingClassifier(
        n_estimators=200, random_state=seed),
}

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

for fs_name, fs_cols in FEATURE_SETS.items():
    feats = resolve_features(fs_cols, panel.columns)
    print(f"\n{'='*60}")
    print(f"Feature set: {fs_name}  ({len(feats)} features)")

    # Fixed split arrays (used by LR and RF, and for final GB test eval)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(panel.loc[mask_train, feats].fillna(0))
    X_val   = scaler.transform(panel.loc[mask_val,   feats].fillna(0))
    X_test  = scaler.transform(panel.loc[mask_test,  feats].fillna(0))
    y_train = panel.loc[mask_train, TARGET].values
    y_val   = panel.loc[mask_val,   TARGET].values
    y_test  = panel.loc[mask_test,  TARGET].values

    for model_name, model_fn in MODELS.items():
        seed_metrics = {k: [] for k in METRIC_COLS}

        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)

            # ==============================================================
            # GRADIENT BOOSTING: walk-forward cross-validation
            # Each fold trains on its expanding window, evaluates on the
            # next window. Final test-set evaluation uses a model trained
            # on the full train+val period with the same seed.
            # ==============================================================
            if model_name == "Gradient Boosting":
                fold_metrics = {k: [] for k in METRIC_COLS}

                for fold_idx, (fold_train_ids, fold_val_ids) in enumerate(cv_folds):
                    fold_tr_mask  = panel["time_id"].isin(fold_train_ids)
                    fold_val_mask = panel["time_id"].isin(fold_val_ids)

                    # Fit scaler on this fold's training window only
                    fold_scaler  = StandardScaler()
                    X_fold_train = fold_scaler.fit_transform(
                        panel.loc[fold_tr_mask, feats].fillna(0))
                    X_fold_val   = fold_scaler.transform(
                        panel.loc[fold_val_mask, feats].fillna(0))
                    y_fold_train = panel.loc[fold_tr_mask,  TARGET].values
                    y_fold_val   = panel.loc[fold_val_mask, TARGET].values

                    clf = model_fn(seed)
                    clf.fit(X_fold_train, y_fold_train)

                    fold_val_prob = clf.predict_proba(X_fold_val)[:, 1]
                    thr           = best_threshold_f1(y_fold_val, fold_val_prob)
                    y_fold_pred   = (fold_val_prob >= thr).astype(int)

                    # Skip fold if val set has only one class (edge case)
                    if len(np.unique(y_fold_val)) < 2:
                        continue

                    m = compute_metrics(y_fold_val, y_fold_pred, fold_val_prob)
                    for k in METRIC_COLS:
                        fold_metrics[k].append(m[k])

                # Average metrics across folds for this seed
                for k in METRIC_COLS:
                    seed_metrics[k].append(
                        float(np.mean(fold_metrics[k])) if fold_metrics[k] else np.nan
                    )

                # Also train a final model on full train+val for test-set
                # evaluation (saved separately so the held-out test is
                # still evaluated cleanly for comparison with other tiers)
                X_trainval = scaler.fit_transform(
                    panel.loc[mask_train | mask_val, feats].fillna(0))
                y_trainval  = panel.loc[mask_train | mask_val, TARGET].values
                X_test_sc   = scaler.transform(
                    panel.loc[mask_test, feats].fillna(0))

                clf_final = model_fn(seed)
                clf_final.fit(X_trainval, y_trainval)
                test_prob = clf_final.predict_proba(X_test_sc)[:, 1]
                # Threshold from last fold's val set (closest to test period)
                thr_final = best_threshold_f1(y_fold_val, fold_val_prob)
                y_pred    = (test_prob >= thr_final).astype(int)

                all_seed_rows.append({
                    "Model": model_name, "Feature Set": fs_name,
                    "Seed": seed, "eval": "test",
                    **{k: compute_metrics(y_test, y_pred, test_prob)[k]
                       for k in METRIC_COLS},
                })

            # ==============================================================
            # LOGISTIC REGRESSION & RANDOM FOREST: fixed split (unchanged)
            # ==============================================================
            else:
                clf = model_fn(seed)
                clf.fit(X_train, y_train)

                val_prob  = clf.predict_proba(X_val)[:, 1]
                test_prob = clf.predict_proba(X_test)[:, 1]

                thr    = best_threshold_f1(y_val, val_prob)
                y_pred = (test_prob >= thr).astype(int)
                m      = compute_metrics(y_test, y_pred, test_prob)

                for k in METRIC_COLS:
                    seed_metrics[k].append(m[k])

                all_seed_rows.append({
                    "Model": model_name, "Feature Set": fs_name,
                    "Seed": seed, "eval": "test",
                    **{k: m[k] for k in METRIC_COLS},
                })

        # Summary row (mean +/- CI across seeds)
        row = {"Model": model_name, "Feature Set": fs_name}
        for k in METRIC_COLS:
            m, h = ci95(seed_metrics[k])
            row[f"{k} mean"] = round(m, 4)
            row[f"{k} CI95"] = round(h, 4)
        summary_rows.append(row)

        ba_m, ba_h   = ci95(seed_metrics["Balanced Accuracy"])
        auc_m, auc_h = ci95(seed_metrics["AUC-ROC"])
        print(f"  {model_name:<22} BA={ba_m:.4f}+-{ba_h:.4f}  "
              f"AUC={auc_m:.4f}+-{auc_h:.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df_seeds   = pd.DataFrame(all_seed_rows)
df_summary = pd.DataFrame(summary_rows)

df_seeds.to_csv(RESULTS_DIR / "tier1_all_seeds.csv",    index=False)
df_summary.to_csv(RESULTS_DIR / "tier1_summary_ci.csv", index=False)

print("\n" + "="*60)
print("TIER 1 SUMMARY (mean +/- 95% CI across 10 seeds)")
print("="*60)
cols_show = ["Model", "Feature Set",
             "Balanced Accuracy mean", "Balanced Accuracy CI95",
             "F1 (positive) mean",     "F1 (positive) CI95",
             "AUC-ROC mean",           "AUC-ROC CI95",
             "PR-AUC mean",            "PR-AUC CI95"]
print(df_summary[cols_show].to_string(index=False))
print(f"\nFull results -> {RESULTS_DIR}")


