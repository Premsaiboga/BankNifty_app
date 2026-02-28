"""
Train Model V2
===============
Trains a Gradient Boosting model with 22 features.
Uses XGBoost if available, falls back to sklearn GradientBoosting.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
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
print(f"Total samples: {len(df)}")
print(f"Win rate: {df['result'].mean()*100:.1f}%")

# =========================
# PREPARE FEATURES
# =========================
# Fill NaN features with defaults
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
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

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
    # Test that xgboost actually works (can fail if libomp missing)
    XGBClassifier()

    print("\nUsing XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=1,
        scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1),
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
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        min_samples_split=50,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    model_type = "sklearn_gb"

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(f"\n{'='*50}")
print("MODEL PERFORMANCE (V2)")
print(f"{'='*50}")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC: {auc:.4f}")

# Per-strategy analysis
print(f"\n{'='*50}")
print("PER-STRATEGY PERFORMANCE")
print(f"{'='*50}")

test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
test_df["actual"] = y_test.values
test_df["pred_prob"] = y_prob

for strat_name, strat_code in STRATEGY_MAP.items():
    mask = test_df["strategy_encoded"] == strat_code
    if mask.sum() == 0:
        continue
    subset = test_df[mask]
    actual_wr = subset["actual"].mean() * 100
    # Find optimal threshold for this strategy
    best_threshold = 0.50
    best_pnl = -999
    for threshold in np.arange(0.40, 0.65, 0.02):
        taken = subset[subset["pred_prob"] >= threshold]
        if len(taken) < 10:
            continue
        wr = taken["actual"].mean()
        rr_val = 1.5 if strat_name in ["EMA_SCALP", "VWAP_REVERSION", "PIVOT_SCALP"] else 2.0
        pnl = taken["actual"].sum() * rr_val - (len(taken) - taken["actual"].sum())
        if pnl > best_pnl:
            best_pnl = pnl
            best_threshold = threshold

    taken = subset[subset["pred_prob"] >= best_threshold]
    filtered_wr = taken["actual"].mean() * 100 if len(taken) > 0 else 0

    print(f"\n{strat_name}:")
    print(f"  Raw samples: {len(subset)}, Win rate: {actual_wr:.1f}%")
    print(f"  Best threshold: {best_threshold:.2f}")
    print(f"  Filtered: {len(taken)} trades, Win rate: {filtered_wr:.1f}%")

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
}

joblib.dump(bundle, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
print(f"Model type: {model_type}")
