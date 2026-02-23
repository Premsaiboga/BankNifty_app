from live.live_engine import process_trade
from strategy.vwap_strategy import VWAPPullbackStrategy
from strategy.pivot_strategy import PivotStrategy
from strategy.abcd_strategy import ABCDStrategy

# ===== BRAINS =====
from brain.market_brain import detect_bias, detect_regime
from brain.strategy_brain import allow_trade
from brain.risk_brain import adjust_targets, reversal_warning
from brain.liquidity_engine import liquidity_block

# ===== LIVE EXIT MANAGER =====
from brain.live_exit_manager import register_trade, update_live_exits

# ===== AI FILTER =====
from ml.ai_filter import ai_filter

import os
from dotenv import load_dotenv
from kiteconnect import KiteTicker
from datetime import time as dt_time
from pathlib import Path
import pandas as pd
import threading
import pytz
import time
import sys

# =========================
# PROJECT PATH
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

IST = pytz.timezone("Asia/Kolkata")

load_dotenv()
API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

instrument_token = 260105

ticks_buffer = []
current_minute = None
minute_candles = []
five_min_candles = []
last_trade_keys = set()

# =========================
# WEBSOCKET
# =========================
def on_ticks(ws, ticks):
    global ticks_buffer
    ticks_buffer.extend(ticks)

def on_connect(ws, response):
    ws.subscribe([instrument_token])
    ws.set_mode(ws.MODE_FULL, [instrument_token])
    print("✅ Connected")

def on_close(ws, code, reason):
    print("❌ Closed:", reason)

# =========================
# CANDLE BUILDERS
# =========================
def build_1min_candle(ticks):
    if not ticks:
        return None

    df = pd.DataFrame(ticks)
    ts = df["exchange_timestamp"].iloc[0]
    ts = ts.replace(tzinfo=pytz.utc).astimezone(IST)

    return {
        "datetime": ts.replace(second=0, microsecond=0),
        "open": float(df["last_price"].iloc[0]),
        "high": float(df["last_price"].max()),
        "low": float(df["last_price"].min()),
        "close": float(df["last_price"].iloc[-1]),
    }

def aggregate_5min():
    last5 = minute_candles[-5:]
    return {
        "datetime": last5[-1]["datetime"],
        "open": last5[0]["open"],
        "high": max(c["high"] for c in last5),
        "low": min(c["low"] for c in last5),
        "close": last5[-1]["close"],
    }

# =========================
# INDICATORS
# =========================
def calculate_atr(df, period=14):
    if len(df) < period + 1:
        return None

    trs = []
    for i in range(1, period + 1):
        curr = df.iloc[-i]
        prev = df.iloc[-i - 1]

        tr = max(
            curr["high"] - curr["low"],
            abs(curr["high"] - prev["close"]),
            abs(curr["low"] - prev["close"]),
        )
        trs.append(tr)

    return sum(trs) / period


def calculate_vwap(df):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return typical.expanding().mean()


def calculate_daily_pivots(df):
    df["date"] = df["datetime"].dt.date
    last_day = df["date"].iloc[-1]

    prev = df[df["date"] < last_day]
    if prev.empty:
        return None, None, None

    ph = prev["high"].max()
    pl = prev["low"].min()
    pc = prev["close"].iloc[-1]

    pivot = (ph + pl + pc) / 3
    r1 = 2 * pivot - pl
    s1 = 2 * pivot - ph

    return pivot, r1, s1

# =========================
# MAIN ENGINE
# =========================
def candle_watcher():
    global ticks_buffer, current_minute

    abcd = ABCDStrategy()
    pivot_strategy = PivotStrategy()
    vwap_strategy = VWAPPullbackStrategy()

    while True:

        if not ticks_buffer:
            time.sleep(1)
            continue

        latest_tick = ticks_buffer[-1]["exchange_timestamp"]
        latest_tick = latest_tick.replace(tzinfo=pytz.utc).astimezone(IST)

        if not (dt_time(9, 15) <= latest_tick.time() <= dt_time(15, 30)):
            time.sleep(1)
            continue

        tick_minute = latest_tick.replace(second=0, microsecond=0)

        if current_minute is None:
            current_minute = tick_minute

        if tick_minute > current_minute:

            candle = build_1min_candle(ticks_buffer)

            if candle:
                minute_candles.append(candle)

                if len(minute_candles) % 5 == 0:

                    five = aggregate_5min()
                    five_min_candles.append(five)

                    df = pd.DataFrame(five_min_candles)
                    df["datetime"] = pd.to_datetime(df["datetime"])

                    df["vwap"] = calculate_vwap(df)
                    df["atr"] = calculate_atr(df)

                    pivot, r1, s1 = calculate_daily_pivots(df)
                    df["pivot"] = pivot or 0
                    df["r1"] = r1 or 0
                    df["s1"] = s1 or 0

                    # ===== UPDATE LIVE TRADES =====
                    update_live_exits(df)

                    # ===== MARKET BRAIN =====
                    bias, _ = detect_bias(df)
                    regime = detect_regime(df)

                    print(f"🧠 Bias={bias} | Regime={regime}")

                    trades = []
                    trades += abcd.generate_trades(df)
                    trades += pivot_strategy.generate_trades(df)
                    trades += vwap_strategy.generate_signals(df)

                    for trade in trades:

                        key = f"{trade['strategy']}_{trade['entry']}_{trade['stoploss']}"
                        if key in last_trade_keys:
                            continue

                        if not allow_trade(trade, bias, regime):
                            continue

                        if liquidity_block(trade, df):
                            print("💧 Liquidity block")
                            continue

                        trade = adjust_targets(trade, df)

                        # features for AI
                        last = df.iloc[-1]
                        trade["features"] = {
                            "vwap_distance": abs(trade["entry"] - last["vwap"]),
                            "candle_size": abs(trade["entry"] - trade["stoploss"]),
                            "atr": last["atr"],
                            "pattern_strength": trade.get("pattern_strength", 0),
                        }

                        ai_result = ai_filter(trade, df, bias, regime)

                        if ai_result["decision"] == "SKIP":
                            print("🚫 AI rejected", ai_result["probability"])
                            continue

                        trade["ai_probability"] = ai_result["probability"]

                        if reversal_warning(df):
                            print("⚠ Reversal Warning")

                        last_trade_keys.add(key)

                        process_trade(trade)
                        register_trade(trade)

            ticks_buffer = []
            current_minute = tick_minute

        time.sleep(1)

# =========================
# START
# =========================
kws = KiteTicker(API_KEY, ACCESS_TOKEN)
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

threading.Thread(target=candle_watcher, daemon=True).start()

print("Starting WebSocket...")
kws.connect()