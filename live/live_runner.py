"""
Live Runner V2 — Production-Grade
====================================
WebSocket tick receiver -> 1m/5m candle builder -> brain filters -> V2 strategies -> AI -> Telegram.

Integrates all brain modules:
  - market_brain: 15-min bias (BUY/SELL/NEUTRAL) + regime (TREND/RANGE)
  - strategy_brain: blocks counter-trend trades, regime-incompatible strategies
  - liquidity_engine: blocks trades during stop hunts / equal highs/lows
  - risk_brain: blocks exhausted moves
  - live_exit_manager: trailing stops, breakeven, exit alerts

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
import json
import traceback
from dotenv import load_dotenv
from collections import deque

from strategy.indicators import calculate_all_indicators
from strategy.orb_strategy import ORBStrategy
from strategy.ema_scalp_strategy import EMAScalpStrategy
from strategy.vwap_reversion_strategy import VWAPReversionStrategy
from strategy.pivot_scalp_strategy import PivotScalpStrategy
from live.live_engine_v2 import process_trade_v2
from live.telegram_alert import send_telegram_alert

# Brain modules
from brain.market_brain import detect_bias, detect_regime, detect_macro_bias
from brain.strategy_brain import allow_trade
from brain.liquidity_engine import liquidity_block
from brain.risk_brain import reversal_warning
from brain.live_exit_manager import register_trade, update_live_exits, active_trades


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
IST_OFFSET = timedelta(hours=5, minutes=30)
SERVER_IS_UTC = abs(_tz_offset) < 3600

def to_ist(ts):
    """Convert timestamp to IST if server is UTC."""
    if SERVER_IS_UTC:
        return ts + IST_OFFSET
    return ts

def now_ist():
    """Get current time in IST regardless of server timezone."""
    return to_ist(datetime.now())


# =========================
# GLOBAL STATE
# =========================
ticks_buffer = []
ticks_lock = threading.Lock()  # Thread safety for tick buffer
current_minute = None

candles_1m = deque(maxlen=200)
candles_5m_raw = []        # Accumulated 5m candle dicts for DataFrame
candles_15m_raw = []       # 15m candles for market_brain
candles_5m_for_15m = []    # Buffer to aggregate 5m -> 15m
signaled_trades = set()    # Avoid duplicate alerts
current_date = None        # Track date for daily reset

# Market state (updated from brain modules)
market_bias = "NEUTRAL"
market_regime = "RANGE"
market_macro_bias = "NEUTRAL"

# History file — stores last 5 trading days of 5m candles
HISTORY_FILE = Path(__file__).resolve().parent / "candle_history.json"
# Max candles to keep: 5 days × 75 candles/day (9:15-15:30 = 6.25hrs × 12 per hr)
MAX_HISTORY_CANDLES = 375


# =========================
# HISTORY: SAVE / LOAD (5 trading days)
# =========================
def save_candle_history(notify=False):
    """Save recent candles to disk for next-day macro_bias."""
    try:
        # Combine existing history with today's candles
        history = load_candle_history_raw()
        today_candles = []
        for c in candles_5m_raw:
            cc = dict(c)
            if isinstance(cc.get("datetime"), datetime):
                cc["datetime"] = cc["datetime"].isoformat()
            today_candles.append(cc)

        # Append today's candles (avoid duplicates by time)
        existing_times = {c["datetime"] for c in history}
        for c in today_candles:
            if c["datetime"] not in existing_times:
                history.append(c)

        # Keep only last MAX_HISTORY_CANDLES
        history = history[-MAX_HISTORY_CANDLES:]

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)

        # Count unique days in history
        days = set()
        for c in history:
            days.add(c["datetime"][:10])
        n_days = len(days)

        print(f"  Saved {len(history)} candles ({n_days} days) to history")

        # Send Telegram notification
        if notify:
            msg = (
                f"💾 <b>CANDLE HISTORY SAVED</b>\n\n"
                f"<b>Candles</b> : {len(history)}\n"
                f"<b>Days</b>    : {n_days}\n"
                f"<b>Time</b>    : {now_ist().strftime('%I:%M %p')} IST\n\n"
                f"<i>Last 5 days data restored for macro trend analysis</i>"
            )
            try:
                send_telegram_alert(msg)
            except Exception:
                pass

    except Exception as e:
        print(f"  History save error: {e}")


def load_candle_history_raw():
    """Load raw candle history from disk."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def fetch_historical_candles():
    """Fetch last 7 calendar days of 5m candles from Zerodha API and save to history."""
    try:
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(ACCESS_TOKEN)

        today = now_ist().date()
        from_date = today - timedelta(days=7)

        print(f"  Fetching 5m candles from Zerodha: {from_date} to {today}")

        candles = kite.historical_data(
            instrument_token=260105,  # BANKNIFTY spot
            from_date=from_date,
            to_date=today,
            interval="5minute"
        )

        if not candles:
            print("  No historical candles returned from API")
            return 0

        # Convert to our format and save
        history = []
        for c in candles:
            dt = pd.to_datetime(c["date"])
            # Only keep market hours (9:15 - 15:30 IST)
            if dt.hour < 9 or (dt.hour == 9 and dt.minute < 15) or dt.hour >= 16:
                continue
            history.append({
                "datetime": dt.isoformat(),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": int(c.get("volume", 0)),
            })

        # Keep only last MAX_HISTORY_CANDLES
        history = history[-MAX_HISTORY_CANDLES:]

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)

        # Count unique days
        days = set()
        for c in history:
            days.add(c["datetime"][:10])

        print(f"  Fetched {len(history)} candles ({len(days)} trading days) from Zerodha API")

        # Send Telegram notification
        msg = (
            f"📊 <b>HISTORICAL DATA LOADED</b>\n\n"
            f"<b>Candles</b> : {len(history)}\n"
            f"<b>Days</b>    : {len(days)}\n"
            f"<b>Source</b>  : Zerodha API\n"
            f"<b>Time</b>    : {now_ist().strftime('%I:%M %p')} IST\n\n"
            f"<i>Last 5 days data restored for macro trend analysis</i>"
        )
        try:
            send_telegram_alert(msg)
        except Exception:
            pass

        return len(history)

    except Exception as e:
        print(f"  Historical fetch error: {e}")
        return 0


def load_candle_history():
    """Load previous days' candles into candles_5m_raw for instant macro_bias.
    If no history file exists, fetch from Zerodha API first."""

    # If no history file, fetch from Zerodha API
    if not HISTORY_FILE.exists():
        print("  No candle history file — fetching from Zerodha API...")
        fetch_historical_candles()

    history = load_candle_history_raw()
    if not history:
        print("  No candle history available — macro_bias starts from scratch")
        return

    today = now_ist().date()
    loaded = 0
    for c in history:
        dt = datetime.fromisoformat(c["datetime"])
        # Only load PREVIOUS days' candles (not today — today will build fresh)
        if dt.date() < today:
            candles_5m_raw.append({
                "datetime": dt,
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c.get("volume", 0),
            })
            loaded += 1

    print(f"  Loaded {loaded} historical candles from previous days")
    if loaded >= 50:
        # Immediately compute macro_bias from history
        update_market_brain()
        print(f"  Instant macro_bias: {market_macro_bias} | bias: {market_bias} | regime: {market_regime}")


# =========================
# V2 STRATEGIES
# =========================
strategies = [
    ORBStrategy(rr=2.0),
    EMAScalpStrategy(rr=2.0),
    VWAPReversionStrategy(rr=2.0),
    PivotScalpStrategy(rr=2.0),
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
# BUILD 1M CANDLE (with volume)
# =========================
def build_1m_candle(ticks):
    clean_ticks = []
    for t in ticks:
        ts = get_tick_time(t)
        if ts:
            clean_ticks.append({
                "time": ts,
                "price": t["last_price"],
                "volume": t.get("volume_traded", 0) or 0,
            })

    if not clean_ticks:
        return None

    df = pd.DataFrame(clean_ticks)
    # volume_traded is cumulative — take max as snapshot at candle close
    volume = int(df["volume"].max()) if df["volume"].max() > 0 else 0

    return {
        "time": df["time"].iloc[0].replace(second=0, microsecond=0),
        "open": float(df["price"].iloc[0]),
        "high": float(df["price"].max()),
        "low": float(df["price"].min()),
        "close": float(df["price"].iloc[-1]),
        "volume": volume,
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
        "volume": sum(c.get("volume", 0) for c in last5),
    }


# =========================
# BUILD 15M FROM 5M CANDLES (for market_brain)
# =========================
def try_build_15m(c5):
    """Aggregate 5m candles into 15m. Returns True if a new 15m candle was built."""
    candles_5m_for_15m.append(c5)

    # Build 15m candle every 3 completed 5m candles
    if len(candles_5m_for_15m) >= 3:
        ist_time = to_ist(candles_5m_for_15m[-1]["time"])
        # Trigger on 15-min IST boundaries
        if ist_time.minute % 15 in (14, 0) or len(candles_5m_for_15m) == 3:
            last3 = candles_5m_for_15m[-3:]
            c15 = {
                "close": last3[-1]["close"],
                "open": last3[0]["open"],
                "high": max(c["high"] for c in last3),
                "low": min(c["low"] for c in last3),
                "volume": sum(c.get("volume", 0) for c in last3),
            }
            candles_15m_raw.append(c15)
            candles_5m_for_15m.clear()
            return True
    return False


# =========================
# UPDATE MARKET BRAIN (bias + regime from 5m data directly)
# =========================
def update_market_brain():
    """Update market bias, macro bias, and regime from 5m candle data."""
    global market_bias, market_regime, market_macro_bias

    if len(candles_5m_raw) < 25:
        market_bias = "NEUTRAL"
        market_regime = "RANGE"
        market_macro_bias = "NEUTRAL"
        return

    df_5m = pd.DataFrame(candles_5m_raw)

    try:
        market_bias, bias_score = detect_bias(df_5m)
        market_regime = detect_regime(df_5m)
        market_macro_bias, macro_score = detect_macro_bias(df_5m)
    except Exception as e:
        print(f"  Brain error: {e}")
        market_bias = "NEUTRAL"
        market_regime = "RANGE"
        market_macro_bias = "NEUTRAL"


# =========================
# DAILY RESET
# =========================
def check_daily_reset():
    """Reset state at start of new trading day. Load history for macro_bias."""
    global current_date, signaled_trades, candles_5m_raw
    global candles_15m_raw, candles_5m_for_15m, market_bias, market_regime, market_macro_bias

    today = now_ist().date()
    if current_date != today:
        if current_date is not None:
            # Save today's candles before clearing
            print(f"  Saving candle history before daily reset...")
            save_candle_history()
            print(f"\n{'='*50}")
            print(f"NEW TRADING DAY: {today}")
            print(f"{'='*50}")
        current_date = today
        signaled_trades.clear()
        candles_5m_raw.clear()
        candles_15m_raw.clear()
        candles_5m_for_15m.clear()
        market_bias = "NEUTRAL"
        market_macro_bias = "NEUTRAL"
        market_regime = "RANGE"

        # Load previous days' candles for instant macro_bias
        load_candle_history()


# =========================
# LUNCH-TIME CHECK
# =========================
def is_lunch_time(ist_time):
    """Lunch break disabled — trade 9:15 to 15:15."""
    return False


# =========================
# EXPIRY DAY CHECK
# =========================
def is_expiry_day(ist_date):
    """BankNifty weekly expiry is Wednesday (weekday=2)."""
    return ist_date.weekday() == 2


# =========================
# PROCESS 5M CANDLE THROUGH BRAIN + V2 STRATEGIES
# =========================
def process_5m_candle(c5):
    """Run brain filters + all V2 strategies on accumulated 5m candle data."""

    # Convert time to IST for indicators
    ist_time = to_ist(c5["time"])
    ist_date = ist_time.date()

    candles_5m_raw.append({
        "datetime": ist_time,
        "open": c5["open"],
        "high": c5["high"],
        "low": c5["low"],
        "close": c5["close"],
        "volume": c5.get("volume", 0),
    })


    # Build 15m candle (kept for compatibility)
    try_build_15m(c5)

    # Update market brain every 5m candle (uses 5m data directly)
    update_market_brain()

    n = len(candles_5m_raw)

    # Need minimum candles for indicators (ATR=14, RSI=14)
    if n < 4:
        print(f"  Warming up: {n} candles (need 4+ to start)")
        return

    # Check expiry day
    expiry_day = is_expiry_day(ist_date)

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
        indicators_str += f" | BIAS={market_bias} | MACRO={market_macro_bias} | REGIME={market_regime}"
        if expiry_day:
            indicators_str += " | EXPIRY DAY"
        print(indicators_str)

        if pd.isna(atr):
            print(f"  Waiting for ATR warmup ({n}/6 candles)")
            return

        # Check exit manager for active trades
        if active_trades:
            try:
                update_live_exits(df)
            except Exception as e:
                print(f"  Exit manager error: {e}")

        # Check for move exhaustion (risk_brain) — warning only, does NOT block
        if len(df) >= 5:
            try:
                if reversal_warning(df):
                    print(f"  (low momentum — exhaustion possible)")
            except Exception:
                pass

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

                            # === BRAIN FILTER 1: Strategy permission ===
                            if not allow_trade(trade, market_bias, market_regime, market_macro_bias):
                                print(f"    -> BLOCKED by strategy_brain "
                                      f"(bias={market_bias}, macro={market_macro_bias}, regime={market_regime})")
                                continue

                            # === BRAIN FILTER 2: Liquidity / stop hunt ===
                            try:
                                if liquidity_block(trade, df):
                                    print(f"    -> BLOCKED by liquidity_engine (stop hunt risk)")
                                    continue
                            except Exception:
                                pass  # Don't block trades if liquidity check fails

                            # === BRAIN FILTER 3: No duplicate/conflicting trades ===
                            skip_trade = False
                            for _k, _t in active_trades.items():
                                if _t["strategy"] != trade["strategy"]:
                                    continue
                                # Same strategy + same direction = duplicate
                                if _t["type"] == trade["type"]:
                                    print(f"    -> SKIPPED (already have active "
                                          f"{trade['strategy']} {trade['type']})")
                                    skip_trade = True
                                    break
                                # Same strategy + opposite direction + active at BE
                                # = trend confirmed, don't fight it
                                if _t.get("breakeven", False):
                                    print(f"    -> SKIPPED (active {trade['strategy']} "
                                          f"{_t['type']} at breakeven — trend confirmed, "
                                          f"not taking opposite {trade['type']})")
                                    skip_trade = True
                                    break
                            if skip_trade:
                                continue

                            # === Mark context ===
                            if expiry_day:
                                trade["is_expiry"] = True
                            trade["regime"] = market_regime
                            trade["macro_bias"] = market_macro_bias

                            # Process through V2 engine (AI filter + telegram)
                            result = process_trade_v2(trade)
                            if result is None:
                                print(f"    -> Filtered out by AI or daily limit")
                            else:
                                print(f"    -> ALERT SENT!")
                                # Register with exit manager
                                try:
                                    register_trade(result)
                                except Exception as e:
                                    print(f"    -> Exit manager register error: {e}")

            except Exception as e:
                print(f"  {strategy.__class__.__name__} error: {e}")

        if signals_found == 0 and not pd.isna(atr):
            print(f"  No signals this candle")

    except Exception as e:
        print(f"  Indicator/strategy error: {e}")
        traceback.print_exc()


# =========================
# CANDLE ENGINE (thread-safe)
# =========================
def candle_engine():
    global current_minute

    while True:
        try:
            snapshot = None

            with ticks_lock:
                if not ticks_buffer:
                    pass  # Will sleep below
                else:
                    last_time = get_tick_time(ticks_buffer[-1])
                    if last_time is None:
                        ticks_buffer.clear()
                        time.sleep(0.2)
                        continue

                    minute = last_time.replace(second=0, microsecond=0)

                    if current_minute is None:
                        current_minute = minute

                    if minute > current_minute:
                        # Take snapshot of ticks and clear buffer
                        snapshot = list(ticks_buffer)
                        ticks_buffer.clear()
                        current_minute = minute

            # Process outside lock
            if snapshot is None:
                time.sleep(0.5)
                continue

            c1 = build_1m_candle(snapshot)

            if c1:
                candles_1m.append(c1)

            if len(candles_1m) >= 5:
                last5 = list(candles_1m)[-5:]
                last_ist = to_ist(last5[-1]["time"])

                # Check if we're at a 5-minute IST boundary
                # 5m candle closes at :04, :09, :14, :19, etc. (0-indexed from :00)
                if (last_ist.minute + 1) % 5 == 0:
                    # Validate consecutive minutes (allow tolerance for slight delays)
                    times = [c["time"] for c in last5]
                    gaps = [(times[i+1] - times[i]).total_seconds() for i in range(4)]
                    consecutive = all(30 <= g <= 150 for g in gaps)

                    if consecutive:
                        c5 = build_5m(last5)

                        ist = to_ist(c5["time"])
                        print(f"\n5M CLOSED: O={c5['open']:.0f} H={c5['high']:.0f} "
                              f"L={c5['low']:.0f} C={c5['close']:.0f} "
                              f"V={c5.get('volume', 0)} "
                              f"@ {ist.strftime('%H:%M')} IST")

                        # Check for daily reset
                        check_daily_reset()

                        # Process through brain + V2 strategies
                        process_5m_candle(c5)
                    else:
                        gap_str = [f"{g:.0f}s" for g in gaps]
                        print(f"  WARN: Non-consecutive 1m candles ({gap_str}), skipping 5m build")

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
    print(f"Brain modules: market_brain, strategy_brain, liquidity_engine, risk_brain, exit_manager")


def on_close(ws, code, reason):
    print(f"WebSocket closed: {code} - {reason}")


def on_error(ws, code, reason):
    print(f"WebSocket error: {code} - {reason}")


def on_ticks(ws, ticks):
    with ticks_lock:
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
print("BankNifty Live Runner V2 — Production")
print(f"Strategies: ORB, EMA_SCALP, VWAP_REVERSION, PIVOT_SCALP")
print(f"Brain: bias + regime + liquidity + risk + exits")
print(f"Server UTC: {SERVER_IS_UTC}")
print(f"Started: {now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST")
print("=" * 50)

# Load previous days' candles for instant macro_bias on startup
print("Loading candle history...")
current_date = now_ist().date()
load_candle_history()


# =========================
# DAILY HISTORY SAVE (8 PM IST)
# =========================
def history_save_scheduler():
    """Background thread: saves candle history at 8 PM IST daily with Telegram notification."""
    saved_today = False
    while True:
        try:
            now = now_ist()
            # Save at 8 PM IST (hour=20)
            if now.hour == 20 and not saved_today:
                print(f"\n[8 PM] Saving candle history...")
                save_candle_history(notify=True)
                saved_today = True
            # Reset flag after midnight
            if now.hour == 0:
                saved_today = False
        except Exception as e:
            print(f"  History scheduler error: {e}")
        time.sleep(60)  # Check every minute


threading.Thread(target=candle_engine, daemon=True).start()
threading.Thread(target=history_save_scheduler, daemon=True).start()

print("Starting WebSocket...")
kws.connect(threaded=True)

while True:
    time.sleep(1)
