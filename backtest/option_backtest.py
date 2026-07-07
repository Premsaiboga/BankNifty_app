"""
Option-aware backtest — the HONEST P&L.
========================================
The existing backtest measures profit in BankNifty INDEX POINTS. But you trade
OPTIONS. This re-prices the exact same trades (from backtest_results_v2.csv) as
ATM weekly options using Black-Scholes, so delta (<1) AND theta (decay) both
apply — then subtracts real Zerodha costs. Result: rupee P&L you'd actually see.

Run:  python3 backtest/option_backtest.py
Assumptions are constants at top — change IV etc. to test sensitivity.
"""
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ---- assumptions (tunable; these are GENEROUS to the system) ----
IV = 0.15                 # annualized implied vol (BankNifty weekly ATM, calm)
RISK_FREE = 0.07
LOT_SIZE = 15
STRIKE_STEP = 100
MAX_HOLD_CANDLES = 60     # same as index backtest
# costs (round trip, per lot)
BROKERAGE = 40            # Zerodha ₹20/order × 2
SLIPPAGE_PER_UNIT = 1.0   # bid/ask, per premium unit, each side
STATUTORY_PCT = 0.0005    # STT+exchange+GST+stamp ≈ 0.05% of premium turnover


# ---- Black-Scholes (stdlib only) ----
def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, sigma, is_call, r=RISK_FREE):
    """European option price. T in years. Near-expiry T→0 gives intrinsic (max theta)."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def next_wed_expiry(dt):
    """Next weekly expiry (Wednesday 15:30 IST). If Wed after 15:30, next week."""
    days = (2 - dt.weekday()) % 7
    exp = (dt + timedelta(days=days)).replace(hour=15, minute=30, second=0, microsecond=0)
    if exp <= dt:
        exp += timedelta(days=7)
    return exp


def years_to_expiry(entry_dt, exp_dt):
    return max((exp_dt - entry_dt).total_seconds(), 0) / (365 * 24 * 3600)


def main():
    trades = pd.read_csv(ROOT / "backtest/backtest_results_v2.csv")
    px = pd.read_csv(ROOT / "data/historical/banknifty_5m.csv")
    px["dt"] = pd.to_datetime(px["datetime"]).dt.tz_localize(None)
    px["day"] = px["dt"].dt.date

    gross = net = 0.0
    wins = losses = matched = 0
    rows = []

    for _, t in trades.iterrows():
        day = pd.to_datetime(t["date"]).date()
        entry_px = float(t["entry"])
        is_buy = t["type"] == "BUY"

        # locate entry candle: same day, close == signal entry price
        cand = px[(px["day"] == day) & (px["close"].sub(entry_px).abs() < 0.05)]
        if cand.empty:
            continue
        i0 = cand.index[0]
        entry_dt = px.loc[i0, "dt"]
        matched += 1

        # ATM option; a BUY signal = buy CE, SELL = buy PE
        strike = round(entry_px / STRIKE_STEP) * STRIKE_STEP
        is_call = is_buy
        exp = next_wed_expiry(entry_dt)

        prem_entry = bs_price(entry_px, strike, years_to_expiry(entry_dt, exp), IV, is_call)

        sl = float(t["stoploss"])
        tgt = float(t["target"])

        # walk index forward until SL/target/EOD, then price the option at exit spot+time
        day_rows = px[(px.index > i0) & (px["day"] == day)].head(MAX_HOLD_CANDLES)
        exit_spot, exit_dt = None, None
        for _, c in day_rows.iterrows():
            if is_buy:
                if c["low"] <= sl:
                    exit_spot, exit_dt = sl, c["dt"]; break
                if c["high"] >= tgt:
                    exit_spot, exit_dt = tgt, c["dt"]; break
            else:
                if c["high"] >= sl:
                    exit_spot, exit_dt = sl, c["dt"]; break
                if c["low"] <= tgt:
                    exit_spot, exit_dt = tgt, c["dt"]; break
        if exit_spot is None:  # EOD exit at last candle close
            last = day_rows.iloc[-1] if len(day_rows) else px.loc[i0]
            exit_spot, exit_dt = last["close"], last["dt"]

        prem_exit = bs_price(exit_spot, strike, years_to_expiry(exit_dt, exp), IV, is_call)

        # rupee P&L per lot
        g = (prem_exit - prem_entry) * LOT_SIZE
        turnover = (prem_entry + prem_exit) * LOT_SIZE
        costs = BROKERAGE + SLIPPAGE_PER_UNIT * LOT_SIZE * 2 + STATUTORY_PCT * turnover
        n = g - costs

        gross += g
        net += n
        wins += n > 0
        losses += n <= 0
        rows.append((str(day), t["strategy"], t["type"], round(prem_entry, 1),
                     round(prem_exit, 1), round(n, 0)))

    print(f"Matched trades   : {matched}/{len(trades)}")
    print(f"Assumed IV       : {IV:.0%} | lot {LOT_SIZE} | costs ~₹{BROKERAGE + SLIPPAGE_PER_UNIT*LOT_SIZE*2:.0f}/trade+taxes")
    print("-" * 48)
    print(f"Win rate (rupees): {wins}/{wins+losses} = {100*wins/max(wins+losses,1):.1f}%")
    print(f"GROSS option P&L : ₹{gross:,.0f}  (before costs)")
    print(f"NET  option P&L  : ₹{net:,.0f}  (after brokerage+slippage+taxes)")
    print(f"Net per trade    : ₹{net/max(matched,1):,.0f}")
    print("-" * 48)
    verdict = "PROFITABLE" if net > 0 else "LOSES MONEY"
    print(f"VERDICT: this system {verdict} when traded as real options.")
    print("\nWorst 5 trades (₹):")
    for r in sorted(rows, key=lambda x: x[-1])[:5]:
        print(f"  {r[0]} {r[1]:14} {r[2]:4} entry={r[3]:>6} exit={r[4]:>6} -> ₹{r[5]:,.0f}")


if __name__ == "__main__":
    main()
