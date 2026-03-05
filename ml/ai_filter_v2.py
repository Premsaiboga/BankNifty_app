"""
AI Filter V2
=============
ML-based trade filter using OOS-optimized thresholds.
Uses 25-feature model with strategy-specific thresholds.

Model AUC is 0.52 — marginal discrimination. Strategy selection
matters more than ML filtering. Only profitable strategies are enabled.
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

    # Use model's own optimal thresholds — but ONLY for enabled strategies
    # Strategies set to 0.90 are intentionally disabled (no OOS edge)
    global STRATEGY_THRESHOLDS
    model_thresholds = bundle.get("optimal_thresholds", {})
    if model_thresholds:
        for strat, thresh in model_thresholds.items():
            if strat in STRATEGY_THRESHOLDS and STRATEGY_THRESHOLDS[strat] < 0.85:
                STRATEGY_THRESHOLDS[strat] = thresh

    print(f"Loaded model_v2 ({bundle.get('model_type', 'unknown')})")
    print(f"Thresholds: {STRATEGY_THRESHOLDS}")


# =========================
# STRATEGY THRESHOLDS
# =========================
# Based on OOS training output. Model will override these with its own
# optimal thresholds when loaded.
#
# PROFITABLE strategies (positive OOS expectancy):
#   PIVOT_SCALP:     48.8% WR at RR 1.5 = +0.22R/trade
#   MOMENTUM_SURGE:  42.9% WR at RR 2.0 = +0.29R/trade
#   ORB:             37.5% WR at RR 2.0 = +0.13R/trade
#
# DISABLED strategies (zero/negative edge):
#   VWAP_REVERSION:  40.5% WR at RR 1.5 = +0.01R (breakeven, loses with costs)
#   EMA_SCALP:       Not enough OOS data to validate
STRATEGY_THRESHOLDS = {
    "ORB": 0.40,
    "MOMENTUM_SURGE": 0.48,
    "PIVOT_SCALP": 0.40,
    # Disabled: set impossibly high threshold (effectively off)
    "VWAP_REVERSION": 0.90,
    "EMA_SCALP": 0.90,
}

# Default for unknown strategies
DEFAULT_THRESHOLD = 0.55


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

    decision = "TAKE" if prob >= threshold else "SKIP"

    # Confidence level
    if prob >= threshold + 0.15:
        confidence = "HIGH"
    elif prob >= threshold + 0.05:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "decision": decision,
        "probability": round(prob, 3),
        "threshold": threshold,
        "confidence": confidence,
    }
