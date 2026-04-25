"""
Live Engine V2 — Production-Grade
====================================
Multi-strategy live trade processor with brain-powered filtering.
Receives trade signals, filters through AI, enforces risk limits,
tracks active trades, and sends Telegram alerts with exit updates.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from ml.ai_filter_v2 import ai_filter_v2
from live.telegram_alert import send_telegram_alert
from config import (
    DAILY_MAX_LOSS_R,
    EXIT_TARGET_R,
    MAX_DAILY_LOSS,
    MAX_CONSECUTIVE_LOSSES,
    MAX_TRADES_PER_DAY,
    TRAIL_BREAKEVEN_R,
    TRAIL_FIXED_TARGET_SL,
    TRAIL_LOCK_PROFIT_R,
    TRAIL_LOCK_TRIGGER_R,
    USE_FIXED_TARGET_EXIT,
)

# =========================
# TIMEZONE
# =========================
from time import timezone as _tz_offset
IST_OFFSET = timedelta(hours=5, minutes=30)
SERVER_IS_UTC = abs(_tz_offset) < 3600

def _now_ist():
    now = datetime.now()
    return now + IST_OFFSET if SERVER_IS_UTC else now


# =========================
# CONFIG
# =========================
MAX_TRADES_EXPIRY = MAX_TRADES_PER_DAY


# =========================
# STATE
# =========================
daily_trades = {
    "date": None,
    "count": 0,
    "trades": [],
    "pnl": 0,
    "pnl_r": 0.0,
    "consecutive_losses": 0,
}
recent_signals = []  # Track correlated signals


def reset_daily_state():
    today = _now_ist().date()
    if daily_trades["date"] != today:
        daily_trades["date"] = today
        daily_trades["count"] = 0
        daily_trades["trades"] = []
        daily_trades["pnl"] = 0
        daily_trades["pnl_r"] = 0.0
        daily_trades["consecutive_losses"] = 0
        recent_signals.clear()


def record_trade_result(pnl_amount: float, trade=None):
    """Called by exit manager when a trade closes. Tracks daily P&L and R."""
    daily_trades["pnl"] += pnl_amount
    if trade:
        risk = abs(trade["entry"] - trade.get("original_sl", trade.get("stoploss", trade["entry"])))
        pnl_r = pnl_amount / risk if risk > 0 else 0
        daily_trades["pnl_r"] += pnl_r
        if pnl_r < 0:
            daily_trades["consecutive_losses"] += 1
        else:
            daily_trades["consecutive_losses"] = 0


# =========================
# TELEGRAM FORMATTERS
# =========================
def format_trade_alert_v2(trade: dict, ai_result: dict) -> str:
    """Format trade alert with the configured exit plan."""

    direction = "🟢 BUY" if trade["type"] == "BUY" else "🔴 SELL"
    entry = trade["entry"]
    sl = trade["stoploss"]
    sl_dist = abs(entry - sl)

    target_r = float(trade.get("exit_target_r", EXIT_TARGET_R))
    if trade["type"] == "BUY":
        fixed_target = entry + sl_dist * target_r
    else:
        fixed_target = entry - sl_dist * target_r

    # Calculate reference levels
    if trade["type"] == "BUY":
        t1 = entry + sl_dist * 1  # 1:1
        t2 = entry + sl_dist * 2  # 1:2
        t3 = entry + sl_dist * 3  # 1:3
        t4 = entry + sl_dist * 4  # 1:4
    else:
        t1 = entry - sl_dist * 1
        t2 = entry - sl_dist * 2
        t3 = entry - sl_dist * 3
        t4 = entry - sl_dist * 4

    now = _now_ist()

    msg = (
        f"📌 <b>BANKNIFTY TRADE ALERT</b>\n\n"
        f"<b>Strategy</b> : {trade['strategy']}\n"
        f"<b>Type</b>     : {direction}\n"
        f"<b>Entry</b>    : {entry:.1f}\n"
        f"<b>SL</b>       : {sl:.1f} ({sl_dist:.0f} pts)\n\n"
        f"<b>Exit Plan:</b>\n"
        f"  Target ({target_r:.1f}R) : {fixed_target:.1f}\n"
        f"  Reference 1R  : {t1:.1f}\n"
        f"  Reference 2R  : {t2:.1f}\n\n"
        f"<b>AI Prob</b>  : {ai_result['probability']}\n"
        f"<b>Time</b>     : {now.strftime('%I:%M %p')}\n\n"
        f"<i>Exit: target {target_r:.1f}R"
        f"{f', trail BE {TRAIL_BREAKEVEN_R:.1f}R, lock {TRAIL_LOCK_PROFIT_R:.1f}R at {TRAIL_LOCK_TRIGGER_R:.1f}R' if TRAIL_FIXED_TARGET_SL else ''}</i>\n"
        f"<i>Regime: {trade.get('regime', 'N/A')} | Macro: {trade.get('macro_bias', 'N/A')}</i>"
    )

    return msg


def format_exit_alert(trade: dict, exit_price: float, reason: str) -> str:
    """Format exit alert with RR achieved for Telegram."""
    entry = trade["entry"]
    trade_type = trade["type"]

    if trade_type == "BUY":
        pnl_pts = exit_price - entry
    else:
        pnl_pts = entry - exit_price

    sl_dist = abs(entry - trade.get("original_sl", trade.get("stoploss", entry)))
    rr_achieved = pnl_pts / sl_dist if sl_dist > 0 else 0

    emoji = "✅" if pnl_pts > 0 else "🔴"
    if rr_achieved >= 2.0:
        emoji = "🏆"  # Big win
    elif rr_achieved >= 1.0:
        emoji = "💰"  # Good win

    now = _now_ist()

    msg = (
        f"{emoji} <b>TRADE EXIT</b>\n\n"
        f"<b>Strategy</b> : {trade['strategy']}\n"
        f"<b>Type</b>     : {trade_type}\n"
        f"<b>Entry</b>    : {entry:.1f}\n"
        f"<b>Exit</b>     : {exit_price:.1f}\n"
        f"<b>P&L</b>      : {pnl_pts:+.1f} pts (<b>{rr_achieved:+.1f}R</b>)\n"
        f"<b>Reason</b>   : {reason}\n"
        f"<b>Time</b>     : {now.strftime('%I:%M %p')}"
    )

    return msg


def format_daily_summary(trades_today: list) -> str:
    """Format end-of-day summary."""
    if not trades_today:
        return "📊 <b>Daily Summary</b>\nNo trades today."

    total = len(trades_today)
    strategies = {}
    for t in trades_today:
        s = t["strategy"]
        strategies[s] = strategies.get(s, 0) + 1

    strat_text = "\n".join(f"  {k}: {v}" for k, v in strategies.items())

    return (
        f"📊 <b>DAILY TRADE SUMMARY</b>\n"
        f"Total Signals: {total}\n"
        f"Strategies:\n{strat_text}\n"
        f"Daily P&L: {daily_trades['pnl']:+.0f} pts"
    )


# =========================
# CORRELATED SIGNAL DETECTION
# =========================
def check_correlated_signals(trade: dict) -> int:
    """Returns count of same-direction signals in last 5 minutes."""
    now = _now_ist()
    trade_type = trade["type"]

    # Clean old signals (> 10 min)
    cutoff = now - timedelta(minutes=10)
    recent_signals[:] = [s for s in recent_signals if s["time"] > cutoff]

    # Count same direction within 5 min
    same_dir = [s for s in recent_signals
                if s["type"] == trade_type
                and (now - s["time"]).total_seconds() < 300]

    # Record this signal
    recent_signals.append({
        "time": now,
        "type": trade_type,
        "strategy": trade["strategy"],
    })

    return len(same_dir)


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
    max_trades = MAX_TRADES_EXPIRY if trade.get("is_expiry") else MAX_TRADES_PER_DAY
    if daily_trades["count"] >= max_trades:
        print(f"[LIMIT] Max {max_trades} trades/day reached. Skipping.")
        return None

    # Daily loss limit
    if daily_trades["pnl"] <= -MAX_DAILY_LOSS:
        print(f"[LOSS LIMIT] Daily loss {daily_trades['pnl']:.0f} exceeds {MAX_DAILY_LOSS}. Stopped.")
        return None

    if daily_trades["pnl_r"] <= -DAILY_MAX_LOSS_R:
        print(f"[R LOSS LIMIT] Daily loss {daily_trades['pnl_r']:.1f}R reached. Stopped.")
        return None

    if daily_trades["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        print(f"[COOLDOWN] {daily_trades['consecutive_losses']} losses in a row. Stopped for today.")
        return None

    # AI Filter — the ONLY quality gate. If AI says TAKE, we trade.
    ai_result = ai_filter_v2(trade)

    regime = trade.get("regime", "RANGE")
    status = f"[{trade['strategy']}] {trade['type']} @ {trade['entry']:.0f}"
    status += f" | AI={ai_result['probability']:.0%} ({ai_result['confidence']})"
    status += f" | {ai_result['decision']} | regime={regime}"
    print(status)

    if ai_result["decision"] != "TAKE":
        return None

    if USE_FIXED_TARGET_EXIT:
        sl_dist = abs(trade["entry"] - trade["stoploss"])
        if trade["type"] == "BUY":
            trade["target"] = round(trade["entry"] + EXIT_TARGET_R * sl_dist, 2)
        else:
            trade["target"] = round(trade["entry"] - EXIT_TARGET_R * sl_dist, 2)
        trade["exit_target_r"] = EXIT_TARGET_R
        trade["rr"] = EXIT_TARGET_R

    # Correlated signal warning
    same_dir_count = check_correlated_signals(trade)
    corr_warning = ""
    if same_dir_count >= 2:
        corr_warning = f"\n\n_Note: {same_dir_count + 1} {trade['type']} signals in 5min — same directional bet_"

    # Send Telegram alert (BankNifty index levels only)
    msg = format_trade_alert_v2(trade, ai_result)
    if corr_warning:
        msg += corr_warning
    send_telegram_alert(msg)

    # Track daily state
    daily_trades["count"] += 1
    daily_trades["trades"].append({
        "strategy": trade["strategy"],
        "type": trade["type"],
        "entry": trade["entry"],
        "time": _now_ist().strftime("%H:%M"),
    })

    print(f"  -> ALERT SENT! Trade #{daily_trades['count']} today")
    return trade


# =========================
# LOCAL TEST
# =========================
if __name__ == "__main__":
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
        "consolidation_ratio": 1.8,
        "ema_spread": 0.3,
        "range_vs_avg": 1.2,
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
