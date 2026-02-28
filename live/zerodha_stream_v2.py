"""
Zerodha WebSocket Streamer V2
==============================
Real-time tick → candle builder with multi-strategy signal generation.

Architecture:
1. Receives tick data via Zerodha WebSocket
2. Builds 1-min candles from ticks
3. Aggregates 5 x 1-min candles into 5-min candles
4. Calculates all indicators on 5-min candles
5. Runs all 5 strategies on each new 5-min candle
6. Sends qualifying trades to live_engine_v2 for AI filtering & alerts

Candle timing:
- 1-min candles: Built on every minute boundary
- 5-min candles: Aggregated at :05, :10, :15, :20, etc.
"""

import sys
from pathlib import Path
import os
import time
import threading
import math
import numpy as np
from datetime import datetime
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from kiteconnect import KiteTicker
from live.live_engine_v2 import process_trade_v2
from ml.features import extract_features, STRATEGY_MAP

# =========================
# ENV
# =========================
API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
INSTRUMENT_TOKEN = 260105  # BANKNIFTY spot

# =========================
# GLOBAL STATE
# =========================
ticks_buffer = []
current_minute = None

# 1-min candles (for building 5-min)
candles_1min = deque(maxlen=100)

# 5-min candles (for strategies)
candles_5min = deque(maxlen=100)

# ORB tracking
orb_state = {"date": None, "high": None, "low": None, "candle_count": 0}

# =========================
# TICK HANDLERS
# =========================
def on_ticks(ws, ticks):
    global ticks_buffer
    ticks_buffer.extend(ticks)


def on_connect(ws, response):
    ws.subscribe([INSTRUMENT_TOKEN])
    ws.set_mode(ws.MODE_FULL, [INSTRUMENT_TOKEN])
    print("Connected to Zerodha WebSocket")


def on_close(ws, code, reason):
    print(f"WebSocket closed: {reason}")


# =========================
# CANDLE BUILDERS
# =========================
def build_1min_candle(ticks):
    """Build a 1-min candle from tick data."""
    if not ticks:
        return None

    prices = [t["last_price"] for t in ticks]
    timestamp = ticks[0].get("exchange_timestamp", datetime.now())

    return {
        "time": timestamp.replace(second=0, microsecond=0),
        "open": prices[0],
        "high": max(prices),
        "low": min(prices),
        "close": prices[-1],
    }


def aggregate_5min(candles_1m: list) -> dict:
    """Aggregate 5 x 1-min candles into 1 x 5-min candle."""
    if not candles_1m:
        return None

    return {
        "time": candles_1m[0]["time"],
        "open": candles_1m[0]["open"],
        "high": max(c["high"] for c in candles_1m),
        "low": min(c["low"] for c in candles_1m),
        "close": candles_1m[-1]["close"],
    }


# =========================
# LIVE INDICATORS (on deque of candle dicts)
# =========================
def calc_ema(candles, period):
    """Calculate EMA from candle deque."""
    if len(candles) < period:
        return None
    prices = [c["close"] for c in candles]
    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calc_rsi(candles, period=14):
    """Calculate RSI from candle deque."""
    if len(candles) < period + 1:
        return None
    gains = []
    losses = []
    prices = [c["close"] for c in candles]
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:]) + 1e-10
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_vwap(candles):
    """Calculate session VWAP from candle deque."""
    if not candles:
        return None
    # Filter to today's candles only
    today = datetime.now().date()
    today_candles = [c for c in candles if c["time"].date() == today]
    if not today_candles:
        return None
    tps = [(c["high"] + c["low"] + c["close"]) / 3 for c in today_candles]
    return sum(tps) / len(tps)


def calc_atr(candles, period=14):
    """Calculate ATR from candle deque."""
    if len(candles) < period + 1:
        return None
    trs = []
    candle_list = list(candles)
    for i in range(-period, 0):
        curr = candle_list[i]
        prev = candle_list[i - 1]
        tr = max(
            curr["high"] - curr["low"],
            abs(curr["high"] - prev["close"]),
            abs(curr["low"] - prev["close"]),
        )
        trs.append(tr)
    return sum(trs) / period


def calc_bb(candles, period=20, std_mult=2):
    """Calculate Bollinger Bands."""
    if len(candles) < period:
        return None, None, None
    prices = [c["close"] for c in list(candles)[-period:]]
    mid = np.mean(prices)
    std = np.std(prices)
    return mid + std_mult * std, mid, mid - std_mult * std


def get_live_indicators(candles_5m):
    """Calculate all indicators from 5-min candle deque."""
    if len(candles_5m) < 21:
        return None

    candle = candles_5m[-1]
    prev = candles_5m[-2]

    vwap = calc_vwap(candles_5m)
    atr = calc_atr(candles_5m)
    ema_9 = calc_ema(candles_5m, 9)
    ema_21 = calc_ema(candles_5m, 21)
    rsi = calc_rsi(candles_5m)
    bb_upper, bb_mid, bb_lower = calc_bb(candles_5m)

    if any(v is None for v in [vwap, atr, ema_9, ema_21, rsi, bb_upper]):
        return None

    prev_ema_9 = calc_ema(list(candles_5m)[:-1], 9)
    prev_ema_21 = calc_ema(list(candles_5m)[:-1], 21)

    candle_range = candle["high"] - candle["low"]
    body = abs(candle["close"] - candle["open"])
    atr_safe = max(atr, 0.01)

    # Day high/low from today's candles
    today = datetime.now().date()
    today_candles = [c for c in candles_5m if c["time"].date() == today]
    day_high = max(c["high"] for c in today_candles) if today_candles else candle["high"]
    day_low = min(c["low"] for c in today_candles) if today_candles else candle["low"]

    # Previous candle features
    prev_range = prev["high"] - prev["low"]
    prev_body = abs(prev["close"] - prev["open"])

    # Consecutive direction
    direction = 1 if candle["close"] > candle["open"] else -1
    consec = direction
    for i in range(len(candles_5m) - 2, max(len(candles_5m) - 6, -1), -1):
        c = candles_5m[i]
        d = 1 if c["close"] > c["open"] else -1
        if d == direction:
            consec += direction
        else:
            break

    return {
        "vwap": vwap,
        "atr": atr,
        "ema_9": ema_9,
        "ema_21": ema_21,
        "prev_ema_9": prev_ema_9,
        "prev_ema_21": prev_ema_21,
        "rsi_14": rsi,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "candle": candle,
        "prev": prev,
        "body_ratio": body / (candle_range + 1e-10),
        "upper_wick": (candle["high"] - max(candle["open"], candle["close"])) / (candle_range + 1e-10),
        "lower_wick": (min(candle["open"], candle["close"]) - candle["low"]) / (candle_range + 1e-10),
        "candle_size_norm": candle_range / atr_safe,
        "prev_body_ratio": prev_body / (prev_range + 1e-10),
        "prev_candle_size_norm": prev_range / atr_safe,
        "ema_9_dist": (candle["close"] - ema_9) / atr_safe,
        "ema_21_dist": (candle["close"] - ema_21) / atr_safe,
        "vwap_dist": (candle["close"] - vwap) / atr_safe,
        "bb_position": (candle["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10),
        "bb_width": (bb_upper - bb_lower) / atr_safe,
        "ema_cross": 1 if ema_9 > ema_21 else -1,
        "day_range_position": (candle["close"] - day_low) / (day_high - day_low + 1e-10),
        "consecutive_dir": consec,
        "day_high": day_high,
        "day_low": day_low,
    }


# =========================
# SIGNAL GENERATORS (live versions)
# =========================
def check_orb_signal(candle, indicators):
    """Check for ORB breakout signal."""
    global orb_state

    today = datetime.now().date()

    # Reset ORB for new day
    if orb_state["date"] != today:
        orb_state = {"date": today, "high": None, "low": None, "candle_count": 0, "traded": set()}

    orb_state["candle_count"] += 1

    # First 3 candles build the ORB
    if orb_state["candle_count"] <= 3:
        if orb_state["high"] is None:
            orb_state["high"] = candle["high"]
            orb_state["low"] = candle["low"]
        else:
            orb_state["high"] = max(orb_state["high"], candle["high"])
            orb_state["low"] = min(orb_state["low"], candle["low"])
        return None

    orb_high = orb_state["high"]
    orb_low = orb_state["low"]
    orb_range = orb_high - orb_low

    if orb_range < 30 or orb_range > 300:
        return None

    # BUY breakout
    if (
        "BUY" not in orb_state.get("traded", set())
        and candle["close"] > orb_high
        and candle["close"] > candle["open"]
        and indicators["body_ratio"] > 0.4
    ):
        entry = candle["close"]
        sl = max((orb_high + orb_low) / 2, candle["low"])
        if entry <= sl:
            sl = entry - orb_range * 0.3
        if entry > sl:
            rr = 2.0
            target = entry + (entry - sl) * rr
            orb_state.setdefault("traded", set()).add("BUY")
            return _build_trade("ORB", "BUY", entry, sl, target, rr, candle, indicators)

    # SELL breakout
    if (
        "SELL" not in orb_state.get("traded", set())
        and candle["close"] < orb_low
        and candle["close"] < candle["open"]
        and indicators["body_ratio"] > 0.4
    ):
        entry = candle["close"]
        sl = min((orb_high + orb_low) / 2, candle["high"])
        if entry >= sl:
            sl = entry + orb_range * 0.3
        if entry < sl:
            rr = 2.0
            target = entry - (sl - entry) * rr
            orb_state.setdefault("traded", set()).add("SELL")
            return _build_trade("ORB", "SELL", entry, sl, target, rr, candle, indicators)

    return None


def check_ema_scalp_signal(candle, prev, indicators):
    """Check for EMA crossover scalp signal."""
    if indicators["prev_ema_9"] is None:
        return None

    atr = indicators["atr"]

    # BUY: EMA cross up
    crossed_up = (indicators["prev_ema_9"] <= indicators["prev_ema_21"]
                  and indicators["ema_9"] > indicators["ema_21"])
    if (
        crossed_up
        and indicators["rsi_14"] > 48
        and candle["close"] > indicators["vwap"]
        and candle["close"] > candle["open"]
    ):
        entry = candle["close"]
        sl = max(indicators["ema_21"], candle["low"]) - 5
        if entry > sl:
            sl_dist = entry - sl
            if 0.3 * atr <= sl_dist <= 1.5 * atr:
                rr = 1.5
                target = entry + sl_dist * rr
                return _build_trade("EMA_SCALP", "BUY", entry, sl, target, rr, candle, indicators)

    # SELL: EMA cross down
    crossed_down = (indicators["prev_ema_9"] >= indicators["prev_ema_21"]
                    and indicators["ema_9"] < indicators["ema_21"])
    if (
        crossed_down
        and indicators["rsi_14"] < 52
        and candle["close"] < indicators["vwap"]
        and candle["close"] < candle["open"]
    ):
        entry = candle["close"]
        sl = min(indicators["ema_21"], candle["high"]) + 5
        if entry < sl:
            sl_dist = sl - entry
            if 0.3 * atr <= sl_dist <= 1.5 * atr:
                rr = 1.5
                target = entry - sl_dist * rr
                return _build_trade("EMA_SCALP", "SELL", entry, sl, target, rr, candle, indicators)

    return None


def check_vwap_reversion_signal(candle, prev, indicators):
    """Check for VWAP mean reversion signal."""
    atr = indicators["atr"]
    vwap = indicators["vwap"]

    # BUY: Price below VWAP, bullish reversal
    deviation_down = vwap - candle["close"]
    if (
        deviation_down > 0.4 * atr
        and candle["close"] > candle["open"]
        and indicators["body_ratio"] > 0.35
        and prev["close"] < vwap
    ):
        entry = candle["close"]
        sl = min(candle["low"], prev["low"]) - 3
        sl_dist = entry - sl
        if sl_dist <= 0 or sl_dist > 1.2 * atr:
            sl = entry - 0.8 * atr
            sl_dist = entry - sl
        if sl_dist > 0.2 * atr:
            rr = 1.5
            target = entry + sl_dist * rr
            return _build_trade("VWAP_REVERSION", "BUY", entry, sl, target, rr, candle, indicators)

    # SELL: Price above VWAP, bearish reversal
    deviation_up = candle["close"] - vwap
    if (
        deviation_up > 0.4 * atr
        and candle["close"] < candle["open"]
        and indicators["body_ratio"] > 0.35
        and prev["close"] > vwap
    ):
        entry = candle["close"]
        sl = max(candle["high"], prev["high"]) + 3
        sl_dist = sl - entry
        if sl_dist <= 0 or sl_dist > 1.2 * atr:
            sl = entry + 0.8 * atr
            sl_dist = sl - entry
        if sl_dist > 0.2 * atr:
            rr = 1.5
            target = entry - sl_dist * rr
            return _build_trade("VWAP_REVERSION", "SELL", entry, sl, target, rr, candle, indicators)

    return None


def check_momentum_surge_signal(candle, indicators):
    """Check for momentum surge signal."""
    atr = indicators["atr"]

    is_surge = (
        indicators["body_ratio"] > 0.55
        and indicators["candle_size_norm"] > 0.7
    )
    if not is_surge:
        return None

    # Bullish surge
    if (
        candle["close"] > candle["open"]
        and indicators["rsi_14"] > 50
        and candle["close"] > indicators["vwap"]
    ):
        entry = candle["close"]
        sl = candle["low"] - 3
        sl_dist = entry - sl
        if sl_dist > 0 and sl_dist <= 2.0 * atr:
            rr = 2.0
            target = entry + sl_dist * rr
            return _build_trade("MOMENTUM_SURGE", "BUY", entry, sl, target, rr, candle, indicators)

    # Bearish surge
    elif (
        candle["close"] < candle["open"]
        and indicators["rsi_14"] < 50
        and candle["close"] < indicators["vwap"]
    ):
        entry = candle["close"]
        sl = candle["high"] + 3
        sl_dist = sl - entry
        if sl_dist > 0 and sl_dist <= 2.0 * atr:
            rr = 2.0
            target = entry - sl_dist * rr
            return _build_trade("MOMENTUM_SURGE", "SELL", entry, sl, target, rr, candle, indicators)

    return None


def _build_trade(strategy, trade_type, entry, sl, target, rr, candle, indicators):
    """Build a standardized trade dict with features."""
    atr_safe = max(indicators["atr"], 0.01)
    now = datetime.now()

    features = {
        "strategy_encoded": STRATEGY_MAP.get(strategy, 0),
        "rsi_14": indicators["rsi_14"],
        "ema_9_dist": indicators["ema_9_dist"],
        "ema_21_dist": indicators["ema_21_dist"],
        "ema_cross": indicators["ema_cross"],
        "vwap_dist": indicators["vwap_dist"],
        "bb_position": indicators["bb_position"],
        "bb_width": indicators["bb_width"],
        "body_ratio": indicators["body_ratio"],
        "upper_wick": indicators["upper_wick"],
        "lower_wick": indicators["lower_wick"],
        "candle_size_norm": indicators["candle_size_norm"],
        "prev_body_ratio": indicators["prev_body_ratio"],
        "prev_candle_size_norm": indicators["prev_candle_size_norm"],
        "consecutive_dir": indicators["consecutive_dir"],
        "day_range_position": indicators["day_range_position"],
        "minutes_from_open": (now.hour - 9) * 60 + (now.minute - 15),
        "hour": now.hour,
        "volatility_regime": 1,  # Default medium
        "atr": indicators["atr"],
        "rr": rr,
        "sl_distance_norm": abs(entry - sl) / atr_safe,
    }

    return {
        "strategy": strategy,
        "type": trade_type,
        "entry": round(entry, 2),
        "stoploss": round(sl, 2),
        "target": round(target, 2),
        "rr": rr,
        "features": features,
    }


# =========================
# 5-MIN CANDLE PROCESSOR
# =========================
def on_new_5min_candle(candle):
    """Called when a new 5-min candle is complete. Runs all strategies."""
    candles_5min.append(candle)

    indicators = get_live_indicators(candles_5min)
    if indicators is None:
        print(f"  Warming up indicators... ({len(candles_5min)}/21 candles)")
        return

    now = datetime.now()
    minutes = (now.hour - 9) * 60 + (now.minute - 15)

    # Only trade between 9:30 and 14:45
    if minutes < 15 or minutes > 330:
        return

    print(f"\n{'='*50}")
    print(f"5-min Candle | {candle['time']} | Close: {candle['close']:.0f}")
    print(f"VWAP: {indicators['vwap']:.0f} | ATR: {indicators['atr']:.1f} | RSI: {indicators['rsi_14']:.1f}")
    print(f"EMA9: {indicators['ema_9']:.0f} | EMA21: {indicators['ema_21']:.0f}")
    print(f"{'='*50}")

    prev = indicators["prev"]

    # Check all strategies
    signals = [
        check_orb_signal(candle, indicators),
        check_ema_scalp_signal(candle, prev, indicators),
        check_vwap_reversion_signal(candle, prev, indicators),
        check_momentum_surge_signal(candle, indicators),
    ]

    for signal in signals:
        if signal is not None:
            process_trade_v2(signal)


# =========================
# MINUTE WATCHER
# =========================
def candle_watcher():
    """Main loop: builds 1-min candles and aggregates to 5-min."""
    global ticks_buffer, current_minute

    min_buffer = []  # Buffer for 1-min candles within current 5-min window

    while True:
        now = datetime.now().replace(second=0, microsecond=0)

        if current_minute is None:
            current_minute = now

        if now > current_minute:
            # Build 1-min candle
            candle_1m = build_1min_candle(ticks_buffer)

            if candle_1m:
                candles_1min.append(candle_1m)
                min_buffer.append(candle_1m)

                print(f"  1m | {candle_1m['time'].strftime('%H:%M')} | "
                      f"C={candle_1m['close']:.0f} | "
                      f"H={candle_1m['high']:.0f} L={candle_1m['low']:.0f}")

                # Check if we have 5 minutes worth of candles
                # Aggregate on 5-minute boundaries (:00, :05, :10, etc.)
                if now.minute % 5 == 0 and len(min_buffer) >= 1:
                    candle_5m = aggregate_5min(min_buffer)
                    if candle_5m:
                        on_new_5min_candle(candle_5m)
                    min_buffer = []

            ticks_buffer = []
            current_minute = now

        time.sleep(1)


# =========================
# START
# =========================
def start_stream():
    """Start the WebSocket stream and candle watcher."""
    print("\n" + "="*60)
    print("  BANKNIFTY TRADING SYSTEM V2")
    print("  Strategies: ORB | EMA Scalp | VWAP Reversion | Momentum")
    print("  RR: 1.5-2.0 | AI Filter: Active")
    print("="*60 + "\n")

    kws = KiteTicker(API_KEY, ACCESS_TOKEN)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close

    kws.connect(threaded=True)
    threading.Thread(target=candle_watcher, daemon=True).start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    start_stream()
