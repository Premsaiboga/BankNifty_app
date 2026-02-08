import joblib
import pandas as pd
from pathlib import Path

from config import STRATEGY_THRESHOLDS

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
# AI FILTER
# =========================
def ai_filter(trade):
    """
    trade → standardized trade dict
    returns → decision + probability
    """

    # -------------------------
    # Build feature row (EXACT MATCH)
    # -------------------------
    row = {
        "strategy": STRATEGY_MAP[trade["strategy"]],
        "vwap_distance": trade["features"]["vwap_distance"],
        "candle_size": trade["features"]["candle_size"],
        "atr": trade["features"]["atr"],
        "pattern_strength": trade["features"]["pattern_strength"],
        "rr": trade["rr"],
    }

    df = pd.DataFrame([row])[FEATURES]

    # -------------------------
    # Scale & Predict
    # -------------------------
    X = scaler.transform(df)
    prob = model.predict_proba(X)[0][1]

    # -------------------------
    # Strategy-wise threshold
    # -------------------------
    threshold = STRATEGY_THRESHOLDS[trade["strategy"]]
    decision = "TAKE" if prob >= threshold else "SKIP"

    return {
        "decision": decision,
        "probability": round(prob, 3),
        "threshold": threshold
    }
