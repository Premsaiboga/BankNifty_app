"""
AI Filter V3
=============
ML-based trade filter for V3 pullback-entry strategies.
All 5 strategies enabled with 1:2 RR.
Model will be retrained on V3 pullback data for better discrimination.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = Path(__file__).resolve().parent / "model_v2.pkl"

bundle = None
model = None
scaler = None
FEATURES = None

def _load_model():
    global bundle, model, scaler, FEATURES
    if bundle is not None:
        return

    if not MODEL_PATH.exists():
        print(f"WARNING: Model not found at {MODEL_PATH}")
        print("Run: python ml/build_training_data_v2.py && python ml/train_model_v2.py")
        return

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    FEATURES = bundle["features"]

    # Use model's own optimal thresholds
    global STRATEGY_THRESHOLDS
    model_thresholds = bundle.get("optimal_thresholds", {})
    if model_thresholds:
        for strat, thresh in model_thresholds.items():
            if strat in STRATEGY_THRESHOLDS:
                STRATEGY_THRESHOLDS[strat] = thresh

    print(f"Loaded model_v2 ({bundle.get('model_type', 'unknown')})")
    print(f"Thresholds: {STRATEGY_THRESHOLDS}")


# =========================
# STRATEGY THRESHOLDS
# =========================
# All 5 strategies enabled with V3 pullback entries.
# Model will override with optimal thresholds from training.
STRATEGY_THRESHOLDS = {
    "ORB": 0.45,
    "MOMENTUM_SURGE": 0.45,
    "PIVOT_SCALP": 0.45,
    "VWAP_REVERSION": 0.45,
    "EMA_SCALP": 0.45,
}

# Default for unknown strategies
DEFAULT_THRESHOLD = 0.50


# =========================
# AI FILTER
# =========================
def ai_filter_v2(trade: dict) -> dict:
    """
    Filter trade through ML model.

    trade must contain:
        - strategy: str (strategy name)
        - features: dict (all 25 ML features)
        - rr: float

    Returns:
        - decision: "TAKE" or "SKIP"
        - probability: float (0-1)
        - threshold: float
        - confidence: str ("HIGH", "MEDIUM", "LOW")
    """
    _load_model()

    # If model not loaded, allow all trades (degrade gracefully)
    if model is None:
        return {
            "decision": "TAKE",
            "probability": 0.55,
            "threshold": 0.50,
            "confidence": "NO_MODEL",
        }

    # Build feature row
    features = trade["features"]
    row = {}
    for feat in FEATURES:
        val = features.get(feat, 0)
        # Handle NaN/None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        row[feat] = val

    df = pd.DataFrame([row])[FEATURES]

    # Scale & predict
    X = scaler.transform(df)
    prob = float(model.predict_proba(X)[0][1])

    # Strategy-specific threshold
    strategy = trade.get("strategy", "UNKNOWN")
    threshold = STRATEGY_THRESHOLDS.get(strategy, DEFAULT_THRESHOLD)

    # Confidence level
    if prob >= threshold + 0.15:
        confidence = "HIGH"
    elif prob >= threshold + 0.05:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ONLY take HIGH confidence trades (86.7% profitable in backtest)
    # MEDIUM/LOW confidence trades have negative expectancy
    decision = "TAKE" if confidence == "HIGH" else "SKIP"

    return {
        "decision": decision,
        "probability": round(prob, 3),
        "threshold": threshold,
        "confidence": confidence,
    }
