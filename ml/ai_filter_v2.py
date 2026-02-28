"""
AI Filter V2
=============
Enhanced ML-based trade filter with lower thresholds for messy markets.
Uses 22-feature model for better trade selection.
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
    print(f"Loaded model_v2 ({bundle.get('model_type', 'unknown')})")


# =========================
# STRATEGY THRESHOLDS (LOWERED for messy markets)
# =========================
# These are intentionally lower than v1 to allow more trades through.
# The strategies themselves are better designed, so lower thresholds still
# produce quality trades. The key insight: in messy markets, even a 50-55%
# win rate with RR 1.5-2.0 is profitable.
STRATEGY_THRESHOLDS = {
    "ORB": 0.45,            # Optimal: 0.44 (high RR compensates)
    "EMA_SCALP": 0.42,      # Optimal: 0.40
    "VWAP_REVERSION": 0.42,  # Optimal: 0.40
    "MOMENTUM_SURGE": 0.44,  # Optimal: 0.44
    "PIVOT_SCALP": 0.48,    # Optimal: 0.48
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
        - features: dict (all 22 ML features)
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
