"""
Live Runner V2
===============
WebSocket tick receiver -> 1m/5m candle builder -> V2 multi-strategy engine.

Replaces V1 first_pullback_strategy with 5 V2 strategies:
  ORB, EMA_SCALP, VWAP_REVERSION, MOMENTUM_SURGE, PIVOT_SCALP

Usage (Oracle VM):
  nohup python3 -u -m live.live_runner > live/live.log 2>&1 &
"""

import sys
import builtins
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force flush on all prints (critical for nohup logging)
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)
builtins.print = print

from kiteconnect import KiteTicker
from datetime import datetime, timedelta
from time import timezone as _tz_offset
import pandas as pd
import numpy as np
import time
import threading
import os
import traceback
from dotenv import load_dotenv
from collections import deque

from strategy.indicators import calculate_all_indicators
from strategy.orb_strategy import ORBStrategy
from strategy.ema_scalp_strategy import EMAScalpStrategy
from strategy.vwap_reversion_strategy import VWAPReversionStrategy
from strategy.momentum_surge_strategy import MomentumSurgeStrategy
from strategy.pivot_scalp_strategy import PivotScalpStrategy
from live.live_engine_v2 import process_trade_v2
from live.telegram_alert import send_telegram_alert


# =========================
# ENV
# =========================
load_dotenv()

API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

instrument_tokens = [260105]


# =========================
# TIMEZONE HANDLING
# =========================
# Oracle VMs default to UTC. KiteTicker uses datetime.fromtimestamp()
# which returns LOCAL time. On UTC server, timestamps are UTC not IST.
# We must convert to IST for indicator calculations (minutes_from_open).
IST_OFFSET = timedelta(hours=5, minutes=30)
SERVER_IS_UTC = abs(_tz_offset) < 3600

def to_ist(ts):
    """Convert timestamp to IST if server is UTC."""
    if SERVER_IS_UTC:
        return ts + IST_OFFSET
    return ts


# =========================
# GLOBAL STATE
# =========================
ticks_buffer = []
current_minute = None

candles_1m = deque(maxlen=200)
candles_5m_raw = []        # Accumulated 5m candle dicts for DataFrame
signaled_trades = set()    # Avoid duplicate alerts
current_date = None        # Track date for daily reset


# =========================
# V2 STRATEGIES
# =========================
strategies = [
    ORBStrategy(rr=2.0),
    EMAScalpStrategy(rr=1.5),
    VWAPReversionStrategy(rr=1.5),
    MomentumSurgeStrategy(rr=2.0),
    PivotScalpStrategy(rr=1.5),
]


# =========================
# SAFE TIMESTAMP
# =========================
def get_tick_time(tick):
    ts = (
        tick.get("exchange_timestamp")
        or tick.get("timestamp")
        or tick.get("last_trade_time")
    )
    if ts is None:
        return None
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    return ts


# =========================
# BUILD 1M CANDLE
# =========================
def build_1m_candle(ticks):
    clean_ticks = []
    for t in ticks:
        ts = get_tick_time(t)
        if ts:
            clean_ticks.append({"time": ts, "price": t["last_price"]})

    if not clean_ticks:
        return None

    df = pd.DataFrame(clean_ticks)
    return {
        "time": df["time"].iloc[0].replace(second=0, microsecond=0),
        "open": float(df["price"].iloc[0]),
        "high": float(df["price"].max()),
        "low": float(df["price"].min()),
        "close": float(df["price"].iloc[-1]),
    }


# =========================
# BUILD 5M FROM 1M CANDLES
# =========================
def build_5m(last5):
    return {
        "time": last5[-1]["time"],
        "open": last5[0]["open"],
        "high": max(c["high"] for c in last5),
        "low": min(c["low"] for c in last5),
        "close": last5[-1]["close"],
    }


# =========================
# DAILY RESET
# =========================
def check_daily_reset():
    """Reset state at start of new trading day."""
    global current_date, signaled_trades, candles_5m_raw

    today = datetime.now().date()
    if current_date != today:
        if current_date is not None:
            print(f"\n{'='*50}")
            print(f"NEW TRADING DAY: {today}")
            print(f"{'='*50}")
        current_date = today
        signaled_trades.clear()
        candles_5m_raw.clear()


# =========================
# PROCESS 5M CANDLE THROUGH V2 STRATEGIES
# =========================
def process_5m_candle(c5):
    """Run all V2 strategies on the accumulated 5m candle data."""

    # Convert time to IST for indicators
    ist_time = to_ist(c5["time"])

    candles_5m_raw.append({
        "datetime": ist_time,
        "open": c5["open"],
        "high": c5["high"],
        "low": c5["low"],
        "close": c5["close"],
    })

    n = len(candles_5m_raw)

    # Need minimum candles for indicators (ATR=14, RSI=14)
    if n < 4:
        print(f"  Warming up: {n} candles (need 4+ to start)")
        return

    try:
        # Build DataFrame and compute all indicators
        df = pd.DataFrame(candles_5m_raw)
        df = calculate_all_indicators(df)

        latest = df.iloc[-1]
        latest_time = latest["datetime"]

        # Debug: print key indicators
        vwap = latest.get("vwap", np.nan)
        rsi = latest.get("rsi_14", np.nan)
        atr = latest.get("atr", np.nan)
        ema9 = latest.get("ema_9", np.nan)
        ema21 = latest.get("ema_21", np.nan)
        mfo = latest.get("minutes_from_open", np.nan)

        indicators_str = (
            f"  VWAP={vwap:.0f}" if not pd.isna(vwap) else "  VWAP=N/A"
        )
        indicators_str += f" | RSI={rsi:.1f}" if not pd.isna(rsi) else " | RSI=N/A"
        indicators_str += f" | ATR={atr:.1f}" if not pd.isna(atr) else " | ATR=N/A"
        indicators_str += f" | EMA9={ema9:.0f}" if not pd.isna(ema9) else " | EMA9=N/A"
        indicators_str += f" | EMA21={ema21:.0f}" if not pd.isna(ema21) else " | EMA21=N/A"
        indicators_str += f" | MinsOpen={mfo:.0f}" if not pd.isna(mfo) else " | MinsOpen=N/A"
        print(indicators_str)

        if pd.isna(atr):
            print(f"  Waiting for ATR warmup ({n}/15 candles)")
            return

        # Run all 5 strategies
        signals_found = 0
        for strategy in strategies:
            try:
                trades = strategy.generate_trades(df)

                for trade in trades:
                    # Only process trades at the LATEST candle
                    if trade["time"] == latest_time:
                        trade_key = f"{trade['strategy']}_{trade['type']}_{trade['time']}"

                        if trade_key not in signaled_trades:
                            signaled_trades.add(trade_key)
                            signals_found += 1

                            print(f"  SIGNAL: {trade['strategy']} {trade['type']} "
                                  f"@ {trade['entry']:.0f} | "
                                  f"SL={trade['stoploss']:.0f} | "
                                  f"TGT={trade['target']:.0f} | "
                                  f"RR=1:{trade['rr']}")

                            # Process through V2 engine (AI filter + option calc + telegram)
                            result = process_trade_v2(trade)
                            if result is None:
                                print(f"    -> Filtered out by AI or daily limit")
                            else:
                                print(f"    -> ALERT SENT!")

            except Exception as e:
                print(f"  {strategy.__class__.__name__} error: {e}")

        if signals_found == 0 and not pd.isna(atr):
            print(f"  No signals this candle")

    except Exception as e:
        print(f"  Indicator/strategy error: {e}")
        traceback.print_exc()


# =========================
# CANDLE ENGINE
# =========================
def candle_engine():
    global ticks_buffer, current_minute

    while True:
        try:
            if not ticks_buffer:
                time.sleep(0.5)
                continue

            last_time = get_tick_time(ticks_buffer[-1])
            if last_time is None:
                time.sleep(0.2)
                continue

            minute = last_time.replace(second=0, microsecond=0)

            if current_minute is None:
                current_minute = minute

            if minute > current_minute:

                c1 = build_1m_candle(ticks_buffer)

                if c1:
                    candles_1m.append(c1)

                if len(candles_1m) >= 5:

                    last5 = list(candles_1m)[-5:]

                    if last5[-1]["time"].minute % 5 == 0:

                        c5 = build_5m(last5)

                        ist = to_ist(c5["time"])
                        print(f"\n5M CLOSED: O={c5['open']:.0f} H={c5['high']:.0f} "
                              f"L={c5['low']:.0f} C={c5['close']:.0f} "
                              f"@ {ist.strftime('%H:%M')} IST")

                        # Check for daily reset
                        check_daily_reset()

                        # Process through V2 strategies
                        process_5m_candle(c5)

                ticks_buffer.clear()
                current_minute = minute

            time.sleep(0.5)

        except Exception as e:
            print(f"ENGINE ERROR: {e}")
            traceback.print_exc()
            time.sleep(1)


# =========================
# WEBSOCKET
# =========================
kws = KiteTicker(API_KEY, ACCESS_TOKEN)


def on_connect(ws, response):
    ws.subscribe(instrument_tokens)
    ws.set_mode(ws.MODE_FULL, instrument_tokens)
    tz = "UTC -> IST conversion ON" if SERVER_IS_UTC else "IST (native)"
    print(f"Connected & FULL mode enabled")
    print(f"Server timezone: {tz}")
    strat_names = [s.__class__.__name__ for s in strategies]
    print(f"Strategies active: {', '.join(strat_names)}")


def on_close(ws, code, reason):
    print(f"WebSocket closed: {code} - {reason}")


def on_error(ws, code, reason):
    print(f"WebSocket error: {code} - {reason}")


def on_ticks(ws, ticks):
    for t in ticks:
        if "last_price" in t:
            ticks_buffer.append(t)


kws.on_connect = on_connect
kws.on_ticks = on_ticks
kws.on_close = on_close
kws.on_error = on_error


# =========================
# START
# =========================
print("=" * 50)
print("BankNifty Live Runner V2")
print(f"Strategies: ORB, EMA_SCALP, VWAP_REVERSION, MOMENTUM_SURGE, PIVOT_SCALP")
print(f"Server UTC: {SERVER_IS_UTC}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

threading.Thread(target=candle_engine, daemon=True).start()

print("Starting WebSocket...")
kws.connect(threaded=True)

while True:
    time.sleep(1)
