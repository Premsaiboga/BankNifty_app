from kiteconnect import KiteTicker
from datetime import datetime
import pandas as pd
import time
import threading
import os
from dotenv import load_dotenv
from collections import deque

from .live_engine import ai_filter
from .telegram_alert import send_telegram_alert


# =========================
# ENV
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

candles_1m = deque(maxlen=200)
candles_5m = deque(maxlen=200)

TREND_CONFIRMED = False
PULLBACK_SEEN = False
TRADE_TAKEN = False
VWAP_BREAK_COUNT = 0

LIQUIDITY_ZONES = deque(maxlen=5)


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
            clean_ticks.append({
                "time": ts,
                "price": t["last_price"]
            })

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
# BUILD 5M
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
# LIQUIDITY MEMORY
# =========================
def update_liquidity_zone(candle):

    body = abs(candle["close"] - candle["open"])
    range_ = candle["high"] - candle["low"]

    if range_ > 25 and body > range_ * 0.6:
        LIQUIDITY_ZONES.append((candle["low"], candle["high"]))


def inside_liquidity(price):
    for low, high in LIQUIDITY_ZONES:
        if low <= price <= high:
            return True
    return False


# =========================
# FIRST PULLBACK STRATEGY
# =========================
def first_pullback_strategy(c5, vwap):

    global TREND_CONFIRMED, PULLBACK_SEEN
    global TRADE_TAKEN, VWAP_BREAK_COUNT

    if len(candles_5m) < 4:
        return None

    c1, c2, c3 = candles_5m[-3], candles_5m[-2], candles_5m[-1]

    if (
        c3["high"] > c2["high"] > c1["high"]
        and c3["low"] > c2["low"] > c1["low"]
        and c3["close"] > vwap
    ):
        TREND_CONFIRMED = True

    if not TREND_CONFIRMED:
        return None

    body = abs(c3["close"] - c3["open"])
    wick = c3["high"] - max(c3["open"], c3["close"])

    if wick > body * 0.6:
        return None

    if c3["low"] <= vwap or inside_liquidity(c3["close"]):
        PULLBACK_SEEN = True
        return None

    if PULLBACK_SEEN and not TRADE_TAKEN:

        TRADE_TAKEN = True

        return {
            "strategy": "FIRST_PULLBACK",
            "entry": c3["close"],
            "stoploss": c3["low"],
            "rr": 4,
        }

    if c3["close"] < vwap:
        VWAP_BREAK_COUNT += 1
    else:
        VWAP_BREAK_COUNT = 0

    if VWAP_BREAK_COUNT >= 2:
        TREND_CONFIRMED = False
        PULLBACK_SEEN = False
        TRADE_TAKEN = False
        VWAP_BREAK_COUNT = 0

    return None


# =========================
# TELEGRAM
# =========================
def send_trade_alert(trade, decision):

    entry = trade["entry"]
    sl = trade["stoploss"]
    rr = trade["rr"]

    trade_type = "BUY" if entry > sl else "SELL"
    risk = abs(entry - sl)

    target = entry + risk * rr if trade_type == "BUY" else entry - risk * rr

    msg = (
        f"📌 *BANKNIFTY TRADE ALERT*\n\n"
        f"*Strategy* : FIRST_PULLBACK\n"
        f"*Type*     : {trade_type}\n"
        f"*Entry*    : {entry}\n"
        f"*SL*       : {sl}\n"
        f"*Target*   : {round(target,2)} (RR 1:{rr})\n"
        f"*AI Prob*  : {decision['probability']}\n"
        f"*Time*     : {datetime.now().strftime('%I:%M %p')}"
    )

    send_telegram_alert(msg)


# =========================
# ENGINE
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
                        candles_5m.append(c5)

                        update_liquidity_zone(c5)

                        print("✅ 5M CLOSED:", c5)

                        vwap = calculate_vwap(candles_5m)
                        atr = calculate_atr(candles_5m)

                        trade = first_pullback_strategy(c5, vwap)

                        if trade and atr:

                            df = pd.DataFrame(list(candles_5m))
                            decision = ai_filter(trade, df, None, None)

                            if decision["decision"] == "TAKE":
                                send_trade_alert(trade, decision)

                ticks_buffer.clear()
                current_minute = minute

            time.sleep(0.5)

        except Exception as e:
            print("⚠️ ENGINE ERROR:", e)
            time.sleep(1)


# =========================
# WEBSOCKET
# =========================
kws = KiteTicker(API_KEY, ACCESS_TOKEN)


def on_connect(ws, response):
    ws.subscribe(instrument_tokens)
    ws.set_mode(ws.MODE_FULL, instrument_tokens)
    print("✅ Connected & FULL mode enabled")


def on_ticks(ws, ticks):
    for t in ticks:
        if "last_price" in t:
            ticks_buffer.append(t)


kws.on_connect = on_connect
kws.on_ticks = on_ticks


# =========================
# START
# =========================
threading.Thread(target=candle_engine, daemon=True).start()

print("Starting WebSocket...")
kws.connect(threaded=True)

while True:
    time.sleep(1)