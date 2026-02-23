import joblib
import pandas as pd
from pathlib import Path

from config import STRATEGY_THRESHOLDS
from ml.feature_engineering import build_features

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = Path("ml/model.pkl")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
FEATURES = bundle["features"]
STRATEGY_MAP = bundle["strategy_map"]

# =========================
# AI FILTER (V2)
# =========================
def ai_filter(trade, df, bias, regime):
    """
    trade  -> trade dict
    df     -> latest 5m dataframe
    bias   -> BUY / SELL / NEUTRAL
    regime -> TREND / RANGE
    """

    # -------------------------
    # Build market-aware features
    # -------------------------
    features = build_features(trade, df, bias, regime)

    # strategy encoding
    features["strategy"] = STRATEGY_MAP[trade["strategy"]]

    # RR still important
    features["rr"] = trade.get("rr", 4)

     # build dataframe
    X_df = pd.DataFrame([features])

    # ensure all trained columns exist
    for col in FEATURES:
        if col not in X_df.columns:
            X_df[col] = 0

    # reorder exactly like training
    X_df = X_df[FEATURES]

    # -------------------------
    # Scale + Predict
    # -------------------------
    X = scaler.transform(X_df)
    prob = model.predict_proba(X)[0][1]

    # -------------------------
    # Threshold Decision
    # -------------------------
    threshold = STRATEGY_THRESHOLDS[trade["strategy"]]
    decision = "TAKE" if prob >= threshold else "SKIP"

    return {
        "decision": decision,
        "probability": round(float(prob), 3),
        "threshold": threshold,
        "features": features
    }