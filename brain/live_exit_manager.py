# brain/live_exit_manager.py
"""
Active trade management with trailing stops and exit alerts.
- Breakeven move at 1x ATR
- Trailing stop at 2x ATR
- Telegram alerts on SL hit / target hit
- P&L tracking for daily loss limit
"""

import pandas as pd

active_trades = {}


def register_trade(trade):
    """Store active trade when executed."""
    key = f"{trade['strategy']}_{trade['entry']}"
    active_trades[key] = {
        **trade,
        "original_sl": trade["stoploss"],
        "trailed": False,
        "breakeven": False,
    }


def remove_trade(key):
    if key in active_trades:
        del active_trades[key]


def update_live_exits(df):
    """
    Called every new 5m candle.
    Dynamically adjusts SL based on market movement.
    Sends Telegram alerts on exits and trail updates.
    Returns list of (trade, exit_price, pnl_pts) for closed trades.
    """
    from live.telegram_alert import send_telegram_alert
    from live.live_engine_v2 import format_exit_alert, record_trade_result

    if df is None or df.empty:
        return []

    last_row = df.iloc[-1]
    last_price = last_row["close"]
    atr = last_row.get("atr", None)

    if atr is None or pd.isna(atr):
        return []

    to_remove = []
    closed_trades = []

    for key, trade in active_trades.items():
        entry = trade["entry"]
        sl = trade["stoploss"]
        target = trade["target"]
        trade_type = trade["type"]

        move = last_price - entry if trade_type == "BUY" else entry - last_price

        # =========================
        # TARGET HIT
        # =========================
        if trade_type == "BUY" and last_price >= target:
            pnl = target - entry
            msg = format_exit_alert(trade, target, "TARGET HIT")
            send_telegram_alert(msg)
            record_trade_result(pnl)
            closed_trades.append((trade, target, pnl))
            to_remove.append(key)
            print(f"  EXIT: {trade['strategy']} {trade_type} TARGET HIT at {target:.0f} (+{pnl:.0f}pts)")
            continue

        if trade_type == "SELL" and last_price <= target:
            pnl = entry - target
            msg = format_exit_alert(trade, target, "TARGET HIT")
            send_telegram_alert(msg)
            record_trade_result(pnl)
            closed_trades.append((trade, target, pnl))
            to_remove.append(key)
            print(f"  EXIT: {trade['strategy']} {trade_type} TARGET HIT at {target:.0f} (+{pnl:.0f}pts)")
            continue

        # =========================
        # BREAKEVEN MOVE (> 0.8x SL distance in profit)
        # =========================
        sl_dist = abs(entry - trade["original_sl"])
        if sl_dist > 0 and move > 0.8 * sl_dist and not trade.get("breakeven"):
            if trade_type == "BUY":
                new_sl = max(sl, entry + 2)  # Small buffer above entry
            else:
                new_sl = min(sl, entry - 2)

            if new_sl != sl:
                trade["stoploss"] = new_sl
                trade["breakeven"] = True
                print(f"  TRAIL: {trade['strategy']} {trade_type} SL moved to BREAKEVEN ({new_sl:.0f})")

        # =========================
        # TRAILING STOP (> 1.2x SL distance in profit)
        # =========================
        if sl_dist > 0 and move > 1.2 * sl_dist:
            move_r = move / sl_dist
            if trade_type == "BUY":
                new_sl = entry + (move_r - 0.6) * sl_dist
                if new_sl > trade["stoploss"]:
                    trade["stoploss"] = new_sl
                    trade["trailed"] = True
                    print(f"  TRAIL: {trade['strategy']} {trade_type} SL trailed to {new_sl:.0f}")
            else:
                new_sl = entry - (move_r - 0.6) * sl_dist
                if new_sl < trade["stoploss"]:
                    trade["stoploss"] = new_sl
                    trade["trailed"] = True
                    print(f"  TRAIL: {trade['strategy']} {trade_type} SL trailed to {new_sl:.0f}")

        # =========================
        # SL HIT
        # =========================
        if trade_type == "BUY" and last_price <= trade["stoploss"]:
            pnl = trade["stoploss"] - entry
            reason = "TRAILED SL HIT" if trade.get("trailed") else "SL HIT"
            msg = format_exit_alert(trade, trade["stoploss"], reason)
            send_telegram_alert(msg)
            record_trade_result(pnl)
            closed_trades.append((trade, trade["stoploss"], pnl))
            to_remove.append(key)
            print(f"  EXIT: {trade['strategy']} {trade_type} {reason} at {trade['stoploss']:.0f} ({pnl:+.0f}pts)")

        elif trade_type == "SELL" and last_price >= trade["stoploss"]:
            pnl = entry - trade["stoploss"]
            reason = "TRAILED SL HIT" if trade.get("trailed") else "SL HIT"
            msg = format_exit_alert(trade, trade["stoploss"], reason)
            send_telegram_alert(msg)
            record_trade_result(pnl)
            closed_trades.append((trade, trade["stoploss"], pnl))
            to_remove.append(key)
            print(f"  EXIT: {trade['strategy']} {trade_type} {reason} at {trade['stoploss']:.0f} ({pnl:+.0f}pts)")

    for k in to_remove:
        remove_trade(k)

    return closed_trades
