"""
Shared final quality gate for live trading and backtests.

The strategy modules create candidate setups and the ML model scores them.
This module blocks the common BankNifty failure modes that are expensive in
live trading: stacked weak breakouts in chop, wide-stop entries, and VWAP
reversion trades fighting a strong trend without high conviction.
"""

BREAKOUT_STRATEGIES = {"ORB", "EMA_SCALP", "MOMENTUM_SURGE", "PIVOT_SCALP"}
REVERSION_STRATEGIES = {"VWAP_REVERSION"}


def _num(value, default=0.0):
    try:
        if value is None:
            return default
        # NaN is not equal to itself.
        if value != value:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def passes_quality_gate(trade: dict, ai_result: dict) -> tuple[bool, list[str]]:
    """Return (allowed, reasons). Reasons are populated only when blocked."""

    strategy = trade.get("strategy", "")
    trade_type = trade.get("type", "")
    regime = trade.get("regime", "RANGE")
    features = trade.get("features", {}) or {}
    probability = _num(ai_result.get("probability"), 0.0)

    body_ratio = _num(features.get("body_ratio"), 0.5)
    range_vs_avg = _num(features.get("range_vs_avg"), 1.0)
    sl_distance_norm = _num(features.get("sl_distance_norm"), 1.0)
    consolidation = _num(features.get("consolidation_ratio"), 2.0)
    ema_spread = abs(_num(features.get("ema_spread"), 0.0))
    day_position = _num(features.get("day_range_position"), 0.5)
    vwap_dist = _num(features.get("vwap_dist"), 0.0)

    reasons = []

    # Stops wider than roughly two ATR need a huge move just to reach 1.2R.
    # In live BankNifty that often means the bot is late to the move.
    if sl_distance_norm > 2.05:
        reasons.append(f"stop too wide ({sl_distance_norm:.1f} ATR)")

    if strategy in BREAKOUT_STRATEGIES:
        weak_confirmation = body_ratio < 0.28 and range_vs_avg < 0.90
        if weak_confirmation:
            reasons.append("weak candle confirmation")

        if regime == "CHOPPY":
            if probability < 0.65:
                reasons.append("choppy breakout needs 0.65+ AI")
            if consolidation < 1.35:
                reasons.append("tight chop before breakout")

        if regime != "TREND":
            if trade_type == "BUY" and day_position > 0.82 and probability < 0.72:
                reasons.append("buy chase near day high")
            if trade_type == "SELL" and day_position < 0.18 and probability < 0.72:
                reasons.append("sell chase near day low")

        if ema_spread < 0.08 and probability < 0.68:
            reasons.append("flat EMA structure")

    if strategy in REVERSION_STRATEGIES:
        if regime == "TREND" and probability < 0.70:
            reasons.append("trend reversion needs 0.70+ AI")

        # If a reversion candle has already crossed beyond VWAP, avoid chasing
        # unless the model score is very strong.
        if trade_type == "BUY" and vwap_dist > 0.45 and probability < 0.72:
            reasons.append("VWAP buy already overextended")
        if trade_type == "SELL" and vwap_dist < -0.45 and probability < 0.72:
            reasons.append("VWAP sell already overextended")

        if body_ratio < 0.25 and probability < 0.70:
            reasons.append("weak reversion candle")

    return len(reasons) == 0, reasons
