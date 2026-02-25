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

instrument_tokens = [260105]  # BANKNIFTY


# =========================
# GLOBAL STATE
# =========================
ticks_buffer = []
current_minute = None
candles = deque(maxlen=60)


# =========================
# SAFE DATAFRAME BUILDER
# =========================
REQUIRED_COLUMNS = [
    "open","high","low","close",
    "vwap","atr","pivot","bias","regime"
]

def candles_to_df():
    if len(candles) == 0:
        return None

    df = pd.DataFrame(list(candles))

    # 🔥 guarantee columns exist (NO MORE KEYERROR EVER)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    return df


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


def on_close(ws, code, reason):
    print("❌ WebSocket closed:", reason)


def on_error(ws, code, reason):
    print("🚨 WebSocket error:", reason)


# =========================
# CANDLE BUILDER
# =========================
def build_1min_candle(ticks):
    df = pd.DataFrame(ticks)

    return {
        "time": df["exchange_timestamp"].iloc[0].replace(second=0, microsecond=0),
        "open": float(df["last_price"].iloc[0]),
        "high": float(df["last_price"].max()),
        "low": float(df["last_price"].min()),
        "close": float(df["last_price"].iloc[-1]),
    }


# =========================
# INDICATORS
# =========================
def calculate_vwap(candle_list):
    prices = [(c["high"]+c["low"]+c["close"])/3 for c in candle_list]
    return sum(prices)/len(prices)


def calculate_atr(candle_list, period=14):
    if len(candle_list) < period+1:
        return None

    trs=[]
    for i in range(1,period+1):
        curr=candle_list[-i]
        prev=candle_list[-i-1]

        tr=max(
            curr["high"]-curr["low"],
            abs(curr["high"]-prev["close"]),
            abs(curr["low"]-prev["close"]),
        )
        trs.append(tr)

    return sum(trs)/period


# SIMPLE DAILY PIVOT (SAFE DEFAULT)
def calculate_pivot(c):
    return (c["high"] + c["low"] + c["close"]) / 3


# =========================
# STRATEGY
# =========================
def vwap_long_setup(candle, vwap, atr):
    if atr is None:
        return None

    strength = candle["high"] - candle["low"]

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

            temp = list(candles) + [candle]

            vwap = calculate_vwap(temp)
            atr = calculate_atr(temp)

            candle["vwap"] = vwap
            candle["atr"] = atr if atr else 0.0
            candle["pivot"] = calculate_pivot(candle)

            # safe placeholders
            candle["bias"] = 0
            candle["regime"] = 0

            candles.append(candle)

            print("\n🕯 NEW 1-MIN CANDLE")
            print(candle)

            if atr:
                trade = vwap_long_setup(candle, vwap, atr)

                if trade:
                    df = candles_to_df()
                    decision = ai_filter(trade, df, None, None)

                    print("🎯 VWAP SETUP FOUND")
                    print(decision)
                else:
                    print("No setup")
            else:
                print("⏳ ATR warming up...")

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