# brain/live_exit_manager.py
"""
Active trade management with fixed-target exits and optional dynamic trailing.
- Default: fixed target at configured R multiple
- Optional legacy mode: breakeven at 0.8R and dynamic trail at 1.5R+
- TREND REVERSAL detection: auto-exit on 2+ signals, warn on 1
  (strong reversal candle, EMA cross against trade, RSI extreme)
- Telegram alerts on trail updates, warnings, and exits
- P&L tracking for daily loss limit
"""

import pandas as pd
import numpy as np
from config import (
    EXIT_TARGET_R,
    TRAIL_BREAKEVEN_R,
    TRAIL_FIXED_TARGET_SL,
    TRAIL_LOCK_PROFIT_R,
    TRAIL_LOCK_TRIGGER_R,
    USE_FIXED_TARGET_EXIT,
)

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
    Fixed target mode banks planned wins while trailing SL protects open profit.
    Legacy dynamic trailing remains available through config.
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
        trade_type = trade["type"]

        move = last_price - entry if trade_type == "BUY" else entry - last_price
        sl_dist = abs(entry - trade["original_sl"])
        if sl_dist <= 0:
            continue

        move_r = move / sl_dist

        # =========================
        # FIXED TARGET EXIT
        # Conservative candle handling: if SL and target both touch inside the
        # same 5m candle, count SL first because intrabar order is unknown.
        # =========================
        if USE_FIXED_TARGET_EXIT:
            target_r = float(trade.get("exit_target_r", EXIT_TARGET_R))
            if trade_type == "BUY":
                target_price = trade.get("target", entry + target_r * sl_dist)
                sl_hit = last_row["low"] <= trade["stoploss"]
                target_hit = last_row["high"] >= target_price
            else:
                target_price = trade.get("target", entry - target_r * sl_dist)
                sl_hit = last_row["high"] >= trade["stoploss"]
                target_hit = last_row["low"] <= target_price

            if sl_hit:
                exit_price = trade["stoploss"]
                pnl = exit_price - entry if trade_type == "BUY" else entry - exit_price
                msg = format_exit_alert(trade, exit_price, "SL HIT")
                send_telegram_alert(msg)
                record_trade_result(pnl, trade)
                closed_trades.append((trade, exit_price, pnl))
                to_remove.append(key)
                print(f"  EXIT: {trade['strategy']} {trade_type} SL HIT at {exit_price:.0f} ({pnl:+.0f}pts)")
                continue

            if target_hit:
                pnl = target_price - entry if trade_type == "BUY" else entry - target_price
                reason = f"TARGET HIT (+{target_r:.1f}R)"
                msg = format_exit_alert(trade, target_price, reason)
                send_telegram_alert(msg)
                record_trade_result(pnl, trade)
                closed_trades.append((trade, target_price, pnl))
                to_remove.append(key)
                print(f"  TARGET: {trade['strategy']} {trade_type} at {target_price:.0f} ({pnl:+.0f}pts)")
                continue

            if TRAIL_FIXED_TARGET_SL:
                best_price = last_row["high"] if trade_type == "BUY" else last_row["low"]
                favorable_move = best_price - entry if trade_type == "BUY" else entry - best_price
                favorable_r = favorable_move / sl_dist

                if favorable_r >= TRAIL_BREAKEVEN_R and not trade.get("breakeven"):
                    if trade_type == "BUY":
                        new_sl = max(trade["stoploss"], entry + 2)
                    else:
                        new_sl = min(trade["stoploss"], entry - 2)

                    if new_sl != trade["stoploss"]:
                        trade["stoploss"] = new_sl
                        trade["breakeven"] = True
                        be_label = "entry+2" if trade_type == "BUY" else "entry-2"
                        msg = (
                            f"🔒 <b>BREAKEVEN</b>\n"
                            f"{trade['strategy']} {trade_type}\n"
                            f"SL moved to {new_sl:.0f} ({be_label})\n"
                            f"<i>Risk eliminated; target remains {target_price:.0f}</i>"
                        )
                        send_telegram_alert(msg)
                        print(f"  TRAIL: {trade['strategy']} {trade_type} SL → BREAKEVEN ({new_sl:.0f})")

                if favorable_r >= TRAIL_LOCK_TRIGGER_R:
                    if trade_type == "BUY":
                        new_sl = entry + TRAIL_LOCK_PROFIT_R * sl_dist
                        should_update = new_sl > trade["stoploss"]
                    else:
                        new_sl = entry - TRAIL_LOCK_PROFIT_R * sl_dist
                        should_update = new_sl < trade["stoploss"]

                    if should_update:
                        old_sl = trade["stoploss"]
                        trade["stoploss"] = new_sl
                        trade["trailed"] = True
                        print(
                            f"  TRAIL: {trade['strategy']} {trade_type} SL "
                            f"{old_sl:.0f} → {new_sl:.0f} "
                            f"(locking {TRAIL_LOCK_PROFIT_R:.1f}R before target)"
                        )

            continue

        # =========================
        # TARGET LEVEL NOTIFICATIONS (T1=1R, T2=2R, T3=3R)
        # =========================
        crossed_levels = trade.get("crossed_levels", set())
        for level, label in [(1.0, "T1 (1:1)"), (2.0, "T2 (1:2)"), (3.0, "T3 (1:3)"), (4.0, "T4 (1:4)")]:
            if move_r >= level and label not in crossed_levels:
                crossed_levels.add(label)
                trade["crossed_levels"] = crossed_levels
                pnl_pts = move_r * sl_dist
                msg = (
                    f"📈 <b>{label} CROSSED!</b>\n"
                    f"Strategy: {trade['strategy']} {trade_type}\n"
                    f"Entry: {entry:.0f} → Now: {last_price:.0f}\n"
                    f"P&L: +{pnl_pts:.0f} pts ({move_r:.1f}R)\n"
                    f"Trail SL: {trade['stoploss']:.0f}\n"
                    f"<i>Letting winner run...</i>"
                )
                send_telegram_alert(msg)
                print(f"  {label} CROSSED: {trade['strategy']} {trade_type} at {last_price:.0f} ({move_r:.1f}R)")

        # =========================
        # BREAKEVEN MOVE (> 0.8R profit)
        # =========================
        if move > 0.8 * sl_dist and not trade.get("breakeven"):
            if trade_type == "BUY":
                new_sl = max(sl, entry + 2)
            else:
                new_sl = min(sl, entry - 2)

            if new_sl != sl:
                trade["stoploss"] = new_sl
                trade["breakeven"] = True
                be_label = "entry+2" if trade_type == "BUY" else "entry-2"
                msg = (
                    f"🔒 <b>BREAKEVEN</b>\n"
                    f"{trade['strategy']} {trade_type}\n"
                    f"SL moved to {new_sl:.0f} ({be_label})\n"
                    f"<i>Risk eliminated</i>"
                )
                send_telegram_alert(msg)
                print(f"  TRAIL: {trade['strategy']} {trade_type} SL → BREAKEVEN ({new_sl:.0f})")

        # =========================
        # INTERMEDIATE TRAIL (> 1.0R profit)
        # Locks 0.3R profit so T1 winners don't evaporate to breakeven
        # =========================
        if sl_dist > 0 and move > 1.0 * sl_dist and move <= 1.5 * sl_dist:
            if trade_type == "BUY":
                new_sl = entry + 0.3 * sl_dist
                if new_sl > trade["stoploss"]:
                    old_sl = trade["stoploss"]
                    trade["stoploss"] = new_sl
                    trade["trailed"] = True
                    print(f"  TRAIL: {trade['strategy']} {trade_type} SL {old_sl:.0f} → {new_sl:.0f} (locking 0.3R after T1)")
            else:
                new_sl = entry - 0.3 * sl_dist
                if new_sl < trade["stoploss"]:
                    old_sl = trade["stoploss"]
                    trade["stoploss"] = new_sl
                    trade["trailed"] = True
                    print(f"  TRAIL: {trade['strategy']} {trade_type} SL {old_sl:.0f} → {new_sl:.0f} (locking 0.3R after T1)")

        # =========================
        # DYNAMIC TRAILING (> 1.5R profit)
        # Locks (move - 0.5R): at 2R→1.5R, at 3R→2.5R, at 4R→3.5R
        # =========================
        if sl_dist > 0 and move > 1.5 * sl_dist:
            if trade_type == "BUY":
                new_sl = entry + (move_r - 0.5) * sl_dist
                if new_sl > trade["stoploss"]:
                    old_sl = trade["stoploss"]
                    trade["stoploss"] = new_sl
                    trade["trailed"] = True
                    print(f"  TRAIL: {trade['strategy']} {trade_type} SL {old_sl:.0f} → {new_sl:.0f} (locking {move_r-0.5:.1f}R)")
            else:
                new_sl = entry - (move_r - 0.5) * sl_dist
                if new_sl < trade["stoploss"]:
                    old_sl = trade["stoploss"]
                    trade["stoploss"] = new_sl
                    trade["trailed"] = True
                    print(f"  TRAIL: {trade['strategy']} {trade_type} SL {old_sl:.0f} → {new_sl:.0f} (locking {move_r-0.5:.1f}R)")

        # =========================
        # TREND REVERSAL DETECTION — Early exit warning
        # Checks: (1) Strong reversal candle, (2) EMA cross against trade, (3) RSI extreme
        # Only triggers if trade is in profit (don't panic exit at a loss)
        # =========================
        if move > 0 and len(df) >= 3 and key not in to_remove:
            prev_row = df.iloc[-2] if len(df) >= 2 else None
            reversal_signals = 0
            reversal_reasons = []

            # (1) Strong bearish candle against BUY / bullish candle against SELL
            candle_body = abs(last_row["close"] - last_row["open"])
            candle_range = last_row["high"] - last_row["low"]
            body_ratio = candle_body / candle_range if candle_range > 0 else 0

            if trade_type == "BUY":
                # Big bearish engulfing candle
                if (last_row["close"] < last_row["open"]
                        and body_ratio > 0.6
                        and candle_range > 1.2 * atr):
                    reversal_signals += 1
                    reversal_reasons.append("Strong bearish candle")
            else:
                # Big bullish engulfing candle
                if (last_row["close"] > last_row["open"]
                        and body_ratio > 0.6
                        and candle_range > 1.2 * atr):
                    reversal_signals += 1
                    reversal_reasons.append("Strong bullish candle")

            # (2) EMA 9/21 cross against trade direction
            ema9 = last_row.get("ema_9", None)
            ema21 = last_row.get("ema_21", None)
            prev_ema9 = prev_row.get("ema_9", None) if prev_row is not None else None
            prev_ema21 = prev_row.get("ema_21", None) if prev_row is not None else None

            if all(v is not None and not (isinstance(v, float) and np.isnan(v))
                   for v in [ema9, ema21, prev_ema9, prev_ema21]):
                if trade_type == "BUY" and prev_ema9 >= prev_ema21 and ema9 < ema21:
                    reversal_signals += 1
                    reversal_reasons.append("EMA 9/21 bearish cross")
                elif trade_type == "SELL" and prev_ema9 <= prev_ema21 and ema9 > ema21:
                    reversal_signals += 1
                    reversal_reasons.append("EMA 9/21 bullish cross")

            # (3) RSI at extreme against trade
            rsi = last_row.get("rsi_14", None)
            if rsi is not None and not (isinstance(rsi, float) and np.isnan(rsi)):
                if trade_type == "BUY" and rsi > 78:
                    reversal_signals += 1
                    reversal_reasons.append(f"RSI overbought ({rsi:.0f})")
                elif trade_type == "SELL" and rsi < 22:
                    reversal_signals += 1
                    reversal_reasons.append(f"RSI oversold ({rsi:.0f})")

            # 2+ reversal signals = EXIT NOW, 1 signal = WARNING
            if reversal_signals >= 2:
                pnl = move
                pnl_r = move_r
                reason = f"TREND REVERSAL ({', '.join(reversal_reasons)})"
                msg = format_exit_alert(trade, last_price, reason)
                send_telegram_alert(msg)
                record_trade_result(pnl, trade)
                closed_trades.append((trade, last_price, pnl))
                to_remove.append(key)
                print(f"  ⚠️ REVERSAL EXIT: {trade['strategy']} {trade_type} at {last_price:.0f} ({pnl_r:+.1f}R) — {reason}")

            elif reversal_signals == 1:
                # Just warn — don't auto-exit on single signal
                warn_msg = (
                    f"⚠️ <b>TREND WARNING</b>\n"
                    f"{trade['strategy']} {trade_type}\n"
                    f"Signal: {reversal_reasons[0]}\n"
                    f"Current P&L: {move:+.0f} pts ({move_r:+.1f}R)\n"
                    f"Trail SL at: {trade['stoploss']:.0f}\n"
                    f"<i>Watch closely — consider manual exit</i>"
                )
                # Only warn once per signal type
                warned = trade.get("warned_signals", set())
                new_warns = set(reversal_reasons) - warned
                if new_warns:
                    send_telegram_alert(warn_msg)
                    trade["warned_signals"] = warned | new_warns
                    print(f"  ⚠️ WARNING: {trade['strategy']} {trade_type} — {reversal_reasons[0]}")

        # =========================
        # SL HIT
        # =========================
        if trade_type == "BUY" and last_row["low"] <= trade["stoploss"] and key not in to_remove:
            pnl = trade["stoploss"] - entry
            pnl_r = pnl / sl_dist
            reason = f"TRAILED SL HIT ({pnl_r:+.1f}R)" if trade.get("trailed") else "SL HIT"
            msg = format_exit_alert(trade, trade["stoploss"], reason)
            send_telegram_alert(msg)
            record_trade_result(pnl, trade)
            closed_trades.append((trade, trade["stoploss"], pnl))
            to_remove.append(key)
            print(f"  EXIT: {trade['strategy']} {trade_type} {reason} at {trade['stoploss']:.0f} ({pnl:+.0f}pts)")

        elif trade_type == "SELL" and last_row["high"] >= trade["stoploss"] and key not in to_remove:
            pnl = entry - trade["stoploss"]
            pnl_r = pnl / sl_dist
            reason = f"TRAILED SL HIT ({pnl_r:+.1f}R)" if trade.get("trailed") else "SL HIT"
            msg = format_exit_alert(trade, trade["stoploss"], reason)
            send_telegram_alert(msg)
            record_trade_result(pnl, trade)
            closed_trades.append((trade, trade["stoploss"], pnl))
            to_remove.append(key)
            print(f"  EXIT: {trade['strategy']} {trade_type} {reason} at {trade['stoploss']:.0f} ({pnl:+.0f}pts)")

    for k in to_remove:
        remove_trade(k)

    return closed_trades
