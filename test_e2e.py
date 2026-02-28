"""
End-to-End Test
================
Simulates realistic trade signals through the full pipeline:
  Strategy Signal → AI Filter → Option Calculator → Telegram Alert

No live data needed — uses mock candle data based on recent BankNifty levels.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from ml.ai_filter_v2 import ai_filter_v2
from live.option_calculator import get_option_recommendation
from live.live_engine_v2 import format_trade_alert_v2, process_trade_v2
from live.telegram_alert import send_telegram_alert

# =============================================
# TEST 1: Direct Telegram connection test
# =============================================
print("=" * 50)
print("TEST 1: Telegram Connection")
print("=" * 50)

send_telegram_alert("BankNifty V2 System Test - Connection OK")
print()

# =============================================
# TEST 2: Full pipeline — BUY signal (ORB)
# =============================================
print("=" * 50)
print("TEST 2: BUY Signal (ORB Strategy)")
print("=" * 50)

buy_trade = {
    "strategy": "ORB",
    "type": "BUY",
    "entry": 60530.0,
    "stoploss": 60480.0,
    "target": 60630.0,
    "rr": 2.0,
    "features": {
        "strategy_encoded": 0,
        "rsi_14": 58.0,
        "ema_9_dist": 0.35,
        "ema_21_dist": 0.6,
        "ema_cross": 1,
        "vwap_dist": 0.25,
        "bb_position": 0.65,
        "bb_width": 2.1,
        "body_ratio": 0.62,
        "upper_wick": 0.12,
        "lower_wick": 0.08,
        "candle_size_norm": 0.85,
        "prev_body_ratio": 0.55,
        "prev_candle_size_norm": 0.7,
        "consecutive_dir": 2,
        "day_range_position": 0.65,
        "minutes_from_open": 30,
        "hour": 9,
        "volatility_regime": 1,
        "atr": 48.0,
        "rr": 2.0,
        "sl_distance_norm": 1.04,
    },
}

result = process_trade_v2(buy_trade)
if result:
    print("BUY alert sent!")
else:
    print("BUY trade skipped by AI — sending manually for test...")
    opt = get_option_recommendation(buy_trade, 48.0)
    ai_result = {"probability": 0.55, "confidence": "MEDIUM"}
    msg = format_trade_alert_v2(buy_trade, ai_result, opt)
    send_telegram_alert(msg)
    print("Manual BUY alert sent!")

print()

# =============================================
# TEST 3: Full pipeline — SELL signal (VWAP Reversion)
# =============================================
print("=" * 50)
print("TEST 3: SELL Signal (VWAP Reversion)")
print("=" * 50)

sell_trade = {
    "strategy": "VWAP_REVERSION",
    "type": "SELL",
    "entry": 60780.0,
    "stoploss": 60830.0,
    "target": 60705.0,
    "rr": 1.5,
    "features": {
        "strategy_encoded": 2,
        "rsi_14": 42.0,
        "ema_9_dist": -0.3,
        "ema_21_dist": -0.5,
        "ema_cross": -1,
        "vwap_dist": 0.45,
        "bb_position": 0.72,
        "bb_width": 1.8,
        "body_ratio": 0.58,
        "upper_wick": 0.2,
        "lower_wick": 0.05,
        "candle_size_norm": 0.75,
        "prev_body_ratio": 0.48,
        "prev_candle_size_norm": 0.65,
        "consecutive_dir": -1,
        "day_range_position": 0.7,
        "minutes_from_open": 145,
        "hour": 11,
        "volatility_regime": 1,
        "atr": 52.0,
        "rr": 1.5,
        "sl_distance_norm": 0.96,
    },
}

result = process_trade_v2(sell_trade)
if result:
    print("SELL alert sent!")
else:
    print("SELL trade skipped by AI — sending manually for test...")
    opt = get_option_recommendation(sell_trade, 52.0)
    ai_result = {"probability": 0.58, "confidence": "MEDIUM"}
    msg = format_trade_alert_v2(sell_trade, ai_result, opt)
    send_telegram_alert(msg)
    print("Manual SELL alert sent!")

print()

# =============================================
# TEST 4: Momentum Surge BUY
# =============================================
print("=" * 50)
print("TEST 4: BUY Signal (Momentum Surge)")
print("=" * 50)

momentum_trade = {
    "strategy": "MOMENTUM_SURGE",
    "type": "BUY",
    "entry": 60450.0,
    "stoploss": 60390.0,
    "target": 60570.0,
    "rr": 2.0,
    "features": {
        "strategy_encoded": 3,
        "rsi_14": 62.0,
        "ema_9_dist": 0.5,
        "ema_21_dist": 0.8,
        "ema_cross": 1,
        "vwap_dist": 0.4,
        "bb_position": 0.8,
        "bb_width": 2.5,
        "body_ratio": 0.72,
        "upper_wick": 0.05,
        "lower_wick": 0.1,
        "candle_size_norm": 1.1,
        "prev_body_ratio": 0.4,
        "prev_candle_size_norm": 0.6,
        "consecutive_dir": 3,
        "day_range_position": 0.55,
        "minutes_from_open": 90,
        "hour": 10,
        "volatility_regime": 2,
        "atr": 55.0,
        "rr": 2.0,
        "sl_distance_norm": 1.09,
    },
}

result = process_trade_v2(momentum_trade)
if result:
    print("MOMENTUM alert sent!")
else:
    print("MOMENTUM trade skipped by AI — sending manually for test...")
    opt = get_option_recommendation(momentum_trade, 55.0)
    ai_result = {"probability": 0.52, "confidence": "LOW"}
    msg = format_trade_alert_v2(momentum_trade, ai_result, opt)
    send_telegram_alert(msg)
    print("Manual MOMENTUM alert sent!")

print()
print("=" * 50)
print("ALL TESTS COMPLETE — Check your Telegram!")
print("=" * 50)
