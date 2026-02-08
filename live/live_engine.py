import sys
from pathlib import Path
from datetime import datetime
import os

# =========================
# PROJECT PATH
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# =========================
# IMPORTS
# =========================
from ml.ai_filter import ai_filter
from live.telegram_alert import send_telegram_alert
from config import RR

# =========================
# TELEGRAM CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =========================
# ALERT FORMATTER
# =========================
def format_trade_alert(trade, ai_result):
    return f"""
📌 *BANKNIFTY TRADE ALERT*

*Strategy* : {trade["strategy"]}
*Type*     : {trade["type"]}
*Entry*    : {trade["entry"]}
*SL*       : {trade["stoploss"]}
*Target*   : {trade["target"]} (RR 1:{trade["rr"]})
*AI Prob*  : {ai_result["probability"]}
*Time*     : {trade["time"]}
""".strip()

# =========================
# LIVE TRADE HANDLER
# =========================
def process_trade(trade):
    """
    trade dict MUST contain:
    strategy, type, entry, stoploss, rr,
    features: {vwap_distance, candle_size, atr, pattern_strength}
    """

    # -------------------------
    # AI FILTER (SINGLE SOURCE)
    # -------------------------
    ai_result = ai_filter(trade)

    print(
        f"[{trade['strategy']}] {trade['type']} | "
        f"Prob={ai_result['probability']} | {ai_result['decision']}"
    )

    if ai_result["decision"] != "TAKE":
        return

    # -------------------------
    # TARGET CALCULATION
    # -------------------------
    if trade["type"] == "BUY":
        target = trade["entry"] + (trade["entry"] - trade["stoploss"]) * trade["rr"]
    else:
        target = trade["entry"] - (trade["stoploss"] - trade["entry"]) * trade["rr"]

    trade["target"] = round(target, 2)
    trade["time"] = datetime.now().strftime("%I:%M %p")

    # -------------------------
    # TELEGRAM ALERT
    # -------------------------
    msg = format_trade_alert(trade, ai_result)
    send_telegram_alert(msg)

# =========================
# LOCAL TEST
# =========================
if __name__ == "__main__":

    mock_trade = {
        "strategy": "PIVOT",
        "type": "BUY",
        "entry": 60120,
        "stoploss": 60070,
        "rr": RR,
        "features": {
            "vwap_distance": 18.5,
            "candle_size": 22.3,
            "atr": 45.0,
            "pattern_strength": 0.0
        }
    }

    process_trade(mock_trade)
