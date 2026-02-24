from kiteconnect import KiteTicker
from datetime import datetime
import pandas as pd
import time
import threading
import os
from dotenv import load_dotenv
from collections import deque
from .live_engine import ai_filter


# =========================
# LOAD ENV
# =========================
load_dotenv()

API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

instrument_tokens = [260105]


# =========================
# GLOBAL STATE
# =========================
ticks_buffer = []
current_minute = None
candles = deque(maxlen=30)


# =========================
# HELPER
# =========================
def candles_to_df():
    if len(candles) == 0:
        return None
    return pd.DataFrame(list(candles))


# =========================
# TICK HANDLER
# =========================
def on_ticks(ws, ticks):
    global ticks_buffer
    ticks_buffer.extend(ticks)


def on_connect(ws, response):
    print("✅ Websocket connected")
    ws.subscribe(instrument_tokens)
    ws.set_mode(ws.MODE_FULL, instrument_tokens)
    print("✅ Subscription sent")


def on_close(ws, code, reason):
    print("❌ WebSocket closed:", reason)


def on_error(ws, code, reason):
    print("🚨 WebSocket error:", reason)


# =========================
# CANDLE BUILDER
# =========================
def build_1min_candle(ticks):
    if not ticks:
        return None

    df = pd.DataFrame(ticks)

    return {
        "time": df["exchange_timestamp"].iloc[0].replace(second=0, microsecond=0),
        "open": float(df["last_price"].iloc[0]),
        "high": float(df["last_price"].max()),
        "low": float(df["last_price"].min()),
        "close": float(df["last_price"].iloc[-1]),
    }


def candle_size(c):
    return c["high"] - c["low"]


# =========================
# INDICATORS
# =========================
def calculate_vwap(candles):
    prices = [(c["high"] + c["low"] + c["close"]) / 3 for c in candles]
    return sum(prices) / len(prices)


def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return None

    trs = []
    for i in range(1, period + 1):
        curr = candles[-i]
        prev = candles[-i - 1]

        tr = max(
            curr["high"] - curr["low"],
            abs(curr["high"] - prev["close"]),
            abs(curr["low"] - prev["close"]),
        )
        trs.append(tr)

    return sum(trs) / period


# =========================
# STRATEGY
# =========================
def vwap_long_setup(candle, vwap, atr):
    if atr is None:
        return None

    strength = candle_size(candle)

    if candle["close"] > vwap and strength >= 0.6 * atr:
        return {
            "entry": candle["close"],
            "stoploss": candle["low"],
            "rr": 4,
            "atr": atr
        }

    return None


# =========================
# CANDLE WATCHER
# =========================
def candle_watcher():
    global ticks_buffer, current_minute

    while True:

        if not ticks_buffer:
            time.sleep(1)
            continue

        last_tick_time = ticks_buffer[-1]["exchange_timestamp"]
        minute = last_tick_time.replace(second=0, microsecond=0)

        if current_minute is None:
            current_minute = minute

        if minute > current_minute:

            candle = build_1min_candle(ticks_buffer)

            if candle:
                candles.append(candle)

                print("\n🕯 NEW 1-MIN CANDLE")
                print(candle)

                if len(candles) >= 15:
                    vwap = calculate_vwap(candles)
                    atr = calculate_atr(candles)

                    trade = vwap_long_setup(candle, vwap, atr)

                    if trade:
                        df = candles_to_df()

                        if df is None or len(df) < 15:
                            print("⏳ Waiting for enough candles for AI...")
                            continue

                        decision = ai_filter(trade, df, None, None)

                        print("🎯 VWAP SETUP FOUND")
                        print(decision)
                    else:
                        print("No setup")

            ticks_buffer.clear()
            current_minute = minute

        time.sleep(1)


# =========================
# START WEBSOCKET
# =========================
kws = KiteTicker(API_KEY, ACCESS_TOKEN)

kws.on_connect = on_connect
kws.on_ticks = on_ticks
kws.on_close = on_close
kws.on_error = on_error

threading.Thread(target=candle_watcher, daemon=True).start()

print("Starting WebSocket...")
kws.connect(threaded=True)

while True:
    time.sleep(1)