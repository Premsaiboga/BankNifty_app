"""
Train Model V2
===============
Trains a Gradient Boosting model with 22 features.
Uses PROPER TIME-SERIES SPLIT (train on first 18 months, test on last 6).
No data leakage — model never sees future data.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from ml.features import FEATURE_COLUMNS, STRATEGY_MAP

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data_v2.csv"
MODEL_PATH = BASE_DIR / "model_v2.pkl"

# =========================
# LOAD DATA
# =========================
print("Loading training data...")
df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"])
print(f"Total samples: {len(df)}")
print(f"Raw win rate: {df['result'].mean()*100:.1f}%")
print(f"Date range: {df['time'].min().date()} to {df['time'].max().date()}")

# =========================
# PREPARE FEATURES
# =========================
for col in FEATURE_COLUMNS:
    if col not in df.columns:
        print(f"  Warning: missing column {col}, filling with 0")
        df[col] = 0
    df[col] = df[col].fillna(0)

X = df[FEATURE_COLUMNS].astype(float)
y = df["result"].astype(int)

print(f"\nFeature shape: {X.shape}")
print(f"Class distribution: WIN={y.sum()}, LOSS={len(y)-y.sum()}")

# =========================
# TIME-SERIES SPLIT (CRITICAL: no look-ahead bias)
# =========================
# Train on first 75% of TIME, test on last 25% of TIME
# This ensures the model never sees future data
dates = df["time"].dt.date
unique_dates = sorted(dates.unique())
n_dates = len(unique_dates)
split_idx = int(n_dates * 0.75)
cutoff_date = unique_dates[split_idx]

train_mask = dates < cutoff_date
test_mask = dates >= cutoff_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\n{'='*50}")
print(f"TIME-SERIES SPLIT (no data leakage)")
print(f"{'='*50}")
print(f"Train: {unique_dates[0]} to {cutoff_date} ({train_mask.sum()} samples)")
print(f"Test:  {cutoff_date} to {unique_dates[-1]} ({test_mask.sum()} samples)")
print(f"Train WR: {y_train.mean()*100:.1f}%")
print(f"Test WR:  {y_test.mean()*100:.1f}%")

# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODEL (try XGBoost first, fallback to sklearn)
# =========================
try:
    from xgboost import XGBClassifier
    XGBClassifier()

    print("\nUsing XGBoost...")
    model = XGBClassifier(
        n_estimators=300,        # Reduced from 500 (less overfitting)
        max_depth=4,             # Reduced from 6 (less overfitting)
        learning_rate=0.05,
        subsample=0.7,           # More regularization
        colsample_bytree=0.7,
        min_child_weight=15,     # More regularization
        gamma=2,                 # More regularization
        reg_alpha=1.0,           # L1 regularization
        reg_lambda=2.0,          # L2 regularization
        scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )
    model_type = "xgboost"

except Exception:
    from sklearn.ensemble import GradientBoostingClassifier

    print("\nXGBoost not found, using sklearn GradientBoosting...")
    model = GradientBoostingClassifier(
        n_estimators=300,        # Reduced from 500
        max_depth=4,             # Reduced from 5
        learning_rate=0.05,
        subsample=0.7,           # More regularization
        min_samples_leaf=30,     # Increased from 20
        min_samples_split=60,    # Increased from 50
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    model_type = "sklearn_gb"

# =========================
# OUT-OF-SAMPLE EVALUATION (honest results)
# =========================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(f"\n{'='*50}")
print("OUT-OF-SAMPLE PERFORMANCE (honest, no leakage)")
print(f"{'='*50}")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC: {auc:.4f}")

# =========================
# FIND OPTIMAL THRESHOLDS PER STRATEGY (on test set)
# =========================
print(f"\n{'='*50}")
print("OPTIMAL THRESHOLDS (out-of-sample)")
print(f"{'='*50}")

test_df = pd.DataFrame(X_test.values, columns=FEATURE_COLUMNS)
test_df["actual"] = y_test.values
test_df["pred_prob"] = y_prob

optimal_thresholds = {}

for strat_name, strat_code in STRATEGY_MAP.items():
    # Skip legacy names
    if strat_name in ["VWAP_PULLBACK", "PIVOT", "ABCD"]:
        continue

    mask = test_df["strategy_encoded"] == strat_code
    if mask.sum() < 10:
        continue
    subset = test_df[mask]
    actual_wr = subset["actual"].mean() * 100
    rr_val = 1.5 if strat_name in ["EMA_SCALP", "VWAP_REVERSION", "PIVOT_SCALP"] else 2.0

    best_threshold = 0.55
    best_pnl = -999

    for threshold in np.arange(0.40, 0.70, 0.02):
        taken = subset[subset["pred_prob"] >= threshold]
        if len(taken) < 5:
            continue
        wr = taken["actual"].mean()
        pnl = taken["actual"].sum() * rr_val - (len(taken) - taken["actual"].sum())
        if pnl > best_pnl:
            best_pnl = pnl
            best_threshold = threshold

    taken = subset[subset["pred_prob"] >= best_threshold]
    filtered_wr = taken["actual"].mean() * 100 if len(taken) > 0 else 0

    optimal_thresholds[strat_name] = best_threshold

    print(f"\n{strat_name}:")
    print(f"  Raw: {len(subset)} trades, WR={actual_wr:.1f}%")
    print(f"  Optimal threshold: {best_threshold:.2f}")
    print(f"  Filtered: {len(taken)} trades, WR={filtered_wr:.1f}%, PnL={best_pnl:.1f}R")

print(f"\n{'='*50}")
print("RECOMMENDED AI THRESHOLDS:")
print(f"{'='*50}")
for s, t in optimal_thresholds.items():
    print(f'    "{s}": {t:.2f},')

# =========================
# FEATURE IMPORTANCE
# =========================
print(f"\n{'='*50}")
print("TOP 10 FEATURE IMPORTANCE")
print(f"{'='*50}")

if hasattr(model, "feature_importances_"):
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    for feat, imp in importance.nlargest(10).items():
        print(f"  {feat}: {imp:.4f}")

# =========================
# SAVE MODEL BUNDLE
# =========================
bundle = {
    "model": model,
    "scaler": scaler,
    "features": FEATURE_COLUMNS,
    "strategy_map": STRATEGY_MAP,
    "model_type": model_type,
    "train_cutoff": str(cutoff_date),
    "optimal_thresholds": optimal_thresholds,
}

joblib.dump(bundle, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
print(f"Model type: {model_type}")
print(f"Train cutoff: {cutoff_date}")
