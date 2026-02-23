# brain/live_exit_manager.py

active_trades = {}


def register_trade(trade):
    """
    Store active trade when executed
    """
    key = f"{trade['strategy']}_{trade['entry']}"
    active_trades[key] = trade


def remove_trade(key):
    if key in active_trades:
        del active_trades[key]


def update_live_exits(df):
    """
    Called every new candle
    Dynamically adjusts SL based on market movement
    """

    to_remove = []

    for key, trade in active_trades.items():

        last_price = df.iloc[-1]["close"]
        atr = df.iloc[-1]["atr"]

        if atr is None:
            continue

        entry = trade["entry"]
        sl = trade["stoploss"]
        trade_type = trade["type"]

        move = abs(last_price - entry)

        # =========================
        # BREAKEVEN MOVE
        # =========================
        if move > atr:

            if trade_type == "BUY":
                trade["stoploss"] = max(sl, entry)
            else:
                trade["stoploss"] = min(sl, entry)

        # =========================
        # TRAILING STOP
        # =========================
        if move > atr * 2:

            if trade_type == "BUY":
                trade["stoploss"] = last_price - atr
            else:
                trade["stoploss"] = last_price + atr

        # =========================
        # EXIT CONDITIONS
        # =========================
        if trade_type == "BUY":
            if last_price <= trade["stoploss"]:
                print(f"🔴 Exit BUY {trade['strategy']} at {last_price}")
                to_remove.append(key)

        else:
            if last_price >= trade["stoploss"]:
                print(f"🔴 Exit SELL {trade['strategy']} at {last_price}")
                to_remove.append(key)

    for k in to_remove:
        remove_trade(k)