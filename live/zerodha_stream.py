from kiteconnect import KiteTicker
from datetime import datetime
import pandas as pd
import time
import threading
import os
from dotenv import load_dotenv
from collections import deque
from live_engine import ai_filter

# =========================
# LOAD ENV
# =========================
load_dotenv()

API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

instrument_token = [260105]  # BANKNIFTY spot

# =========================
# GLOBAL STATE
# =========================
ticks_buffer = []
current_minute = None

# store last 30 candles
candles = deque(maxlen=30)

# =========================
# TICK HANDLER
# =========================
def on_ticks(ws, ticks):
    global ticks_buffer
    ticks_buffer.extend(ticks)

def on_connect(ws, response):
    print("✅ Websocket connected")

    print("Instrument tokens:", instrument_token)

    ws.subscribe(instrument_token)
    ws.set_mode(ws.MODE_FULL, instrument_token)

    print("✅ Subscription request sent")

def on_close(ws, code, reason):
    print("❌ WebSocket closed:", reason)

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

def candle_size(candle):
    return candle["high"] - candle["low"]

# =========================
# INDICATORS
# =========================
def calculate_vwap(candles):
    prices = [
        (c["high"] + c["low"] + c["close"]) / 3
        for c in candles
    ]
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

    candle_strength = candle_size(candle)

    if candle["close"] > vwap:
        if candle_strength >= 0.6 * atr:
            return {
                "entry": candle["close"],
                "stoploss": candle["low"],
                "rr": 4,
                "vwap_distance": candle["close"] - vwap,
                "candle_size": candle_strength,
                "atr": atr
            }

    return None

# =========================
# MINUTE WATCHER (CORRECT)
# =========================
def candle_watcher():
    global ticks_buffer, current_minute

    while True:
        now = datetime.now().replace(second=0, microsecond=0)

        if current_minute is None:
            current_minute = now

        # RUN ONLY ON NEW MINUTE
        if now > current_minute:
            candle = build_1min_candle(ticks_buffer)

            if candle:
                candles.append(candle)

                if len(candles) >= 15:
                    vwap = calculate_vwap(candles)
                    atr = calculate_atr(candles)
                    size = candle_size(candle)
                    vwap_dist = candle["close"] - vwap

                    trade = vwap_long_setup(candle, vwap, atr)

                    print("\n🕯 New 1-min Candle")
                    print("Time        :", candle["time"])
                    print("Close       :", candle["close"])
                    print("VWAP        :", round(vwap, 2))
                    print("ATR         :", round(atr, 2))
                    print("Candle size :", round(size, 2))
                    print("VWAP dist   :", round(vwap_dist, 2))

                    if trade:
                        decision = ai_filter(trade)
                        print("🎯 VWAP SETUP FOUND")
                        print("AI Decision :", decision["decision"])
                        print("Probability :", decision["probability"])
                    else:
                        print("No VWAP setup")
                else:
                    print("🕯 Candle (warming up):", candle)

            ticks_buffer = []
            current_minute = now

        time.sleep(1)

# =========================
# START EVERYTHING
# =========================
kws = KiteTicker(API_KEY, ACCESS_TOKEN)
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

kws.connect(threaded=True)

threading.Thread(target=candle_watcher, daemon=True).start()

# keep main thread alive
while True:
    time.sleep(1)
