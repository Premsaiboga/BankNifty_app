"""
Comprehensive indicator calculations for BankNifty trading system.
All indicators pre-calculated on DataFrame for strategies and ML features.
"""

import pandas as pd
import numpy as np


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators on a 5-min OHLCV DataFrame.
    Expects columns: datetime, open, high, low, close
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute

    # Minutes from market open (9:15 AM)
    df["minutes_from_open"] = (df["hour"] - 9) * 60 + (df["minute"] - 15)

    # ========== VWAP (Intraday proxy for index) ==========
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (
        df.groupby("date")["tp"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ========== ATR (14-period) ==========
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    df["tr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()

    # ========== EMA 9 and 21 ==========
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

    # ========== RSI 14 ==========
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ========== Bollinger Bands (20, 2) ==========
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # ========== Candle Properties ==========
    df["candle_range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["body_ratio"] = df["body"] / (df["candle_range"] + 1e-10)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["candle_range"] + 1e-10)
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["candle_range"] + 1e-10)
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)

    # ========== Consecutive Direction ==========
    direction = np.where(df["close"] > df["open"], 1, -1)
    consecutive = np.zeros(len(df))
    for i in range(1, len(df)):
        if direction[i] == direction[i - 1]:
            consecutive[i] = consecutive[i - 1] + direction[i]
        else:
            consecutive[i] = direction[i]
    df["consecutive_dir"] = consecutive

    # ========== Daily Pivot Points (Previous Day) ==========
    daily = df.groupby("date").agg(
        daily_high=("high", "max"),
        daily_low=("low", "min"),
        daily_close=("close", "last"),
    )
    daily["pivot"] = (daily["daily_high"] + daily["daily_low"] + daily["daily_close"]) / 3
    daily["r1"] = 2 * daily["pivot"] - daily["daily_low"]
    daily["r2"] = daily["pivot"] + (daily["daily_high"] - daily["daily_low"])
    daily["s1"] = 2 * daily["pivot"] - daily["daily_high"]
    daily["s2"] = daily["pivot"] - (daily["daily_high"] - daily["daily_low"])

    # Shift by 1 day so we use PREVIOUS day's pivots
    daily_shifted = daily[["pivot", "r1", "r2", "s1", "s2"]].shift(1)

    df = df.merge(daily_shifted, left_on="date", right_index=True, how="left")

    # ========== Opening Range (first 3 candles = 15 min for 5-min TF) ==========
    orb_data = {}
    for date, group in df.groupby("date"):
        first_3 = group.head(3)
        orb_data[date] = {
            "orb_high": first_3["high"].max(),
            "orb_low": first_3["low"].min(),
        }
    orb_df = pd.DataFrame.from_dict(orb_data, orient="index")
    df = df.merge(orb_df, left_on="date", right_index=True, how="left")

    # ========== Running Day High/Low ==========
    df["day_high"] = df.groupby("date")["high"].cummax()
    df["day_low"] = df.groupby("date")["low"].cummin()

    # ========== Derived Normalized Features ==========
    atr_safe = df["atr"].replace(0, np.nan).ffill().fillna(1)
    df["ema_9_dist"] = (df["close"] - df["ema_9"]) / atr_safe
    df["ema_21_dist"] = (df["close"] - df["ema_21"]) / atr_safe
    df["vwap_dist"] = (df["close"] - df["vwap"]) / atr_safe
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / atr_safe
    df["candle_size_norm"] = df["candle_range"] / atr_safe
    day_range = df["day_high"] - df["day_low"]
    df["day_range_position"] = (df["close"] - df["day_low"]) / (day_range + 1e-10)

    # ========== Previous Candle Features ==========
    df["prev_body_ratio"] = df["body_ratio"].shift(1)
    df["prev_candle_size_norm"] = df["candle_size_norm"].shift(1)
    df["prev_is_bullish"] = df["is_bullish"].shift(1)

    # ========== Volatility Regime ==========
    df["returns"] = df["close"].pct_change()
    df["volatility_20"] = df["returns"].rolling(20).std() * 100
    df["volatility_regime"] = pd.cut(
        df["volatility_20"],
        bins=[-np.inf, 0.15, 0.30, np.inf],
        labels=[0, 1, 2],  # 0=low, 1=medium, 2=high
    ).astype(float)

    # ========== EMA Cross Signal ==========
    df["ema_cross"] = np.where(df["ema_9"] > df["ema_21"], 1, -1)
    df["ema_cross_change"] = df["ema_cross"].diff().abs()  # 2 means crossover just happened

    return df
