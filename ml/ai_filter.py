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
# SAFE STRATEGY ENCODER
# =========================
def encode_strategy(strategy_name):
    # if strategy exists → use it
    if strategy_name in STRATEGY_MAP:
        return STRATEGY_MAP[strategy_name]

    # fallback to first known strategy
    print(f"⚠ Unknown strategy '{strategy_name}' → using fallback")
    return list(STRATEGY_MAP.values())[0]


# =========================
# AI FILTER
# =========================
def ai_filter(trade, df, bias, regime):

    # -------------------------
    # Build features
    # -------------------------
    features = build_features(trade, df, bias, regime)

    if features is None:
        return {"decision": "SKIP", "reason": "feature_build_failed"}

    # -------------------------
    # SAFE strategy encoding
    # -------------------------
    features["strategy"] = encode_strategy(
        trade.get("strategy", "UNKNOWN")
    )

    features["rr"] = trade.get("rr", 4)

    X_df = pd.DataFrame([features])

    # ensure trained columns exist
    for col in FEATURES:
        if col not in X_df.columns:
            X_df[col] = 0

    X_df = X_df[FEATURES]

    # -------------------------
    # Predict
    # -------------------------
    X = scaler.transform(X_df)
    prob = model.predict_proba(X)[0][1]

    # safe threshold
    strategy_name = trade.get("strategy", None)
    threshold = STRATEGY_THRESHOLDS.get(strategy_name, 0.5)

    decision = "TAKE" if prob >= threshold else "SKIP"

    return {
        "decision": decision,
        "probability": round(float(prob), 3),
        "threshold": threshold,
        "strategy_used": strategy_name,
        "features": features,
    }