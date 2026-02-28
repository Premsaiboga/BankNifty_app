"""
Live Engine V2
===============
Multi-strategy live trade processor with option premium recommendations.
Receives trade signals from all 5 strategies, filters through AI,
calculates option recommendations, and sends Telegram alerts.
"""

import sys
from pathlib import Path
from datetime import datetime
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from ml.ai_filter_v2 import ai_filter_v2
from live.telegram_alert import send_telegram_alert
from live.option_calculator import get_option_recommendation

# =========================
# CONFIG
# =========================
MAX_TRADES_PER_DAY = 5
MAX_DAILY_LOSS = 3000  # Stop trading after ₹3000 loss in a day

# =========================
# STATE
# =========================
daily_trades = {"date": None, "count": 0, "trades": []}


def reset_daily_state():
    today = datetime.now().date()
    if daily_trades["date"] != today:
        daily_trades["date"] = today
        daily_trades["count"] = 0
        daily_trades["trades"] = []


# =========================
# TELEGRAM FORMATTERS
# =========================
def format_trade_alert_v2(trade: dict, ai_result: dict, opt: dict) -> str:
    """Format clean trade alert for Telegram."""

    direction = "BUY" if trade["type"] == "BUY" else "SELL"
    confidence_tag = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low", "NO_MODEL": "N/A"}

    msg = f"""Strategy: {trade['strategy']}
Type: {direction}
Strike Price: BANKNIFTY {opt['expiry']} {opt['strike']}{opt['option_type']}
Entry: {opt['premium_entry']:.0f}
Stoploss: {opt['premium_sl']:.0f}
Target: {opt['premium_target']:.0f}
AI Prob: {ai_result['probability']:.0%} ({confidence_tag.get(ai_result['confidence'], 'Low')})
Time: {datetime.now().strftime('%I:%M %p')}

Alternative: BANKNIFTY {opt['expiry']} {opt['alt_strike']}{opt['alt_option_type']} / {opt['alt_premium_entry']:.0f} / {opt['alt_premium_sl']:.0f} / {opt['alt_premium_target']:.0f}"""

    return msg


def format_daily_summary(trades_today: list) -> str:
    """Format end-of-day summary."""
    if not trades_today:
        return "📊 *Daily Summary*\nNo trades today."

    total = len(trades_today)
    strategies = {}
    for t in trades_today:
        s = t["strategy"]
        strategies[s] = strategies.get(s, 0) + 1

    strat_text = "\n".join(f"  {k}: {v}" for k, v in strategies.items())

    return f"""
📊 *DAILY TRADE SUMMARY*
━━━━━━━━━━━━━━━━━━━━━
Total Signals: {total}
Strategies:
{strat_text}

💡 Review your positions and book profits!
━━━━━━━━━━━━━━━━━━━━━
""".strip()


# =========================
# LIVE TRADE HANDLER
# =========================
def process_trade_v2(trade: dict):
    """
    Process a trade signal through AI filter and send Telegram alert.

    trade dict MUST contain:
        strategy, type, entry, stoploss, target, rr, features (dict)

    Returns the processed trade dict or None if skipped.
    """
    reset_daily_state()

    # Daily trade limit
    if daily_trades["count"] >= MAX_TRADES_PER_DAY:
        print(f"[LIMIT] Max {MAX_TRADES_PER_DAY} trades/day reached. Skipping.")
        return None

    # AI Filter
    ai_result = ai_filter_v2(trade)

    status = f"[{trade['strategy']}] {trade['type']} @ {trade['entry']:.0f}"
    status += f" | AI={ai_result['probability']:.0%} ({ai_result['confidence']})"
    status += f" | {ai_result['decision']}"
    print(status)

    if ai_result["decision"] != "TAKE":
        return None

    # Calculate option recommendation
    atr = trade["features"].get("atr", 50)
    opt = get_option_recommendation(trade, atr)

    # Send Telegram alert
    msg = format_trade_alert_v2(trade, ai_result, opt)
    send_telegram_alert(msg)

    # Track daily state
    daily_trades["count"] += 1
    daily_trades["trades"].append({
        "strategy": trade["strategy"],
        "type": trade["type"],
        "entry": trade["entry"],
        "time": datetime.now().strftime("%H:%M"),
    })

    print(f"  → ALERT SENT! Trade #{daily_trades['count']} today")
    return trade


# =========================
# LOCAL TEST
# =========================
if __name__ == "__main__":
    from ml.features import extract_features

    # Simulate a mock trade with full features
    mock_features = {
        "strategy_encoded": 0,
        "rsi_14": 55.0,
        "ema_9_dist": 0.3,
        "ema_21_dist": 0.5,
        "ema_cross": 1,
        "vwap_dist": 0.2,
        "bb_position": 0.6,
        "bb_width": 2.0,
        "body_ratio": 0.65,
        "upper_wick": 0.1,
        "lower_wick": 0.15,
        "candle_size_norm": 0.8,
        "prev_body_ratio": 0.5,
        "prev_candle_size_norm": 0.7,
        "consecutive_dir": 2,
        "day_range_position": 0.6,
        "minutes_from_open": 75,
        "hour": 10,
        "volatility_regime": 1,
        "atr": 45.0,
        "rr": 2.0,
        "sl_distance_norm": 0.8,
    }

    mock_trade = {
        "strategy": "ORB",
        "type": "BUY",
        "entry": 60530.0,
        "stoploss": 60480.0,
        "target": 60630.0,
        "rr": 2.0,
        "features": mock_features,
    }

    print("Testing live engine v2...")
    result = process_trade_v2(mock_trade)
    if result:
        print("Trade processed successfully!")
    else:
        print("Trade was skipped by AI filter")
