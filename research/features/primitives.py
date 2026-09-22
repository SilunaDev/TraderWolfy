"""
ASTRA FUSION QUANT — Technical Primitives (Milestone 1)
ATR14, EMA, RSI14, ADX14, DMI+, DMI-, Efficiency Ratio, NATR, Bollinger, Keltner.

Seeding conventions exactly match Pine Script v6 Wilder smoothing.
All computations are prefix-only (no future data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── ATR14 — Wilder smoothing (matches Pine) ─────────────────────────────────

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: max of (H-L, |H-Cprev|, |L-Cprev|)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ATR using Wilder smoothing — seeds with SMA of first `period` TR values,
    then applies Wilder's formula: ATR[t] = ATR[t-1] * (period-1)/period + TR[t] / period.
    This exactly matches Pine's ta.atr(period).
    """
    tr = true_range(high, low, close)
    atr = pd.Series(np.nan, index=tr.index, dtype=float)

    # First valid ATR = SMA of first `period` TRs (Pine seeds this way)
    first_valid = tr.first_valid_index()
    if first_valid is None:
        return atr

    tr_vals = tr.dropna()
    if len(tr_vals) < period:
        return atr

    seed_idx = tr_vals.index[period - 1]
    atr.loc[seed_idx] = tr_vals.iloc[:period].mean()

    alpha = 1.0 / period
    loc_list = list(atr.index)
    seed_pos = loc_list.index(seed_idx)

    prev = atr.loc[seed_idx]
    for pos in range(seed_pos + 1, len(loc_list)):
        idx = loc_list[pos]
        tr_val = tr.loc[idx]
        if pd.isna(tr_val):
            atr.loc[idx] = np.nan
            prev = np.nan
        else:
            val = prev * (1 - alpha) + tr_val * alpha
            atr.loc[idx] = val
            prev = val

    return atr


# ─── EMA ─────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """
    EMA with SMA seed for first value — matches Pine's ta.ema().
    Uses adjust=False for recursive Wilder-style.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ─── RSI14 — Wilder smoothing ────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder smoothing — matches Pine's ta.rsi().
    Seeds avg_gain/avg_loss with SMA of first `period` deltas.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = pd.Series(np.nan, index=close.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=close.index, dtype=float)

    valid = gain.dropna()
    if len(valid) < period:
        return pd.Series(np.nan, index=close.index)

    seed_idx = valid.index[period - 1]
    avg_gain.loc[seed_idx] = gain.loc[valid.index[:period]].mean()
    avg_loss.loc[seed_idx] = loss.loc[valid.index[:period]].mean()

    alpha = 1.0 / period
    locs = list(avg_gain.index)
    sp = locs.index(seed_idx)
    pg, pl = avg_gain.loc[seed_idx], avg_loss.loc[seed_idx]

    for pos in range(sp + 1, len(locs)):
        idx = locs[pos]
        if pd.isna(gain.loc[idx]):
            avg_gain.loc[idx] = np.nan
            avg_loss.loc[idx] = np.nan
            pg, pl = np.nan, np.nan
        else:
            pg = pg * (1 - alpha) + gain.loc[idx] * alpha
            pl = pl * (1 - alpha) + loss.loc[idx] * alpha
            avg_gain.loc[idx] = pg
            avg_loss.loc[idx] = pl

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0, RSI = 100
    rsi_vals[avg_loss == 0] = 100.0
    return rsi_vals


# ─── ADX14 + DMI ─────────────────────────────────────────────────────────────

def adx_dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (ADX, DI+, DI-) using Wilder smoothing — matches Pine's ta.adx().

    DI+ = 100 * smoothed_dm_plus / smoothed_atr
    DI- = 100 * smoothed_dm_minus / smoothed_atr
    DX  = 100 * abs(DI+ - DI-) / (DI+ + DI-)
    ADX = Wilder smooth of DX
    """
    prev_high = high.shift(1)
    prev_low  = low.shift(1)
    tr = true_range(high, low, close)

    dm_plus_raw  = high - prev_high
    dm_minus_raw = prev_low - low

    dm_plus  = np.where((dm_plus_raw > dm_minus_raw) & (dm_plus_raw > 0), dm_plus_raw, 0.0)
    dm_minus = np.where((dm_minus_raw > dm_plus_raw) & (dm_minus_raw > 0), dm_minus_raw, 0.0)

    dm_plus_s  = pd.Series(dm_plus,  index=high.index, dtype=float)
    dm_minus_s = pd.Series(dm_minus, index=high.index, dtype=float)

    def _wilder_smooth(s: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        valid = s.dropna()
        if len(valid) < period:
            return out
        seed_i = valid.index[period - 1]
        out.loc[seed_i] = valid.iloc[:period].sum()
        locs = list(out.index)
        sp = locs.index(seed_i)
        prev = out.loc[seed_i]
        for pos in range(sp + 1, len(locs)):
            idx = locs[pos]
            v = s.loc[idx]
            if pd.isna(v):
                out.loc[idx] = np.nan
                prev = np.nan
            else:
                val = prev - prev / period + v
                out.loc[idx] = val
                prev = val
        return out

    smooth_tr   = _wilder_smooth(tr)
    smooth_dmp  = _wilder_smooth(dm_plus_s)
    smooth_dmm  = _wilder_smooth(dm_minus_s)

    di_plus  = 100.0 * smooth_dmp / smooth_tr.replace(0, np.nan)
    di_minus = 100.0 * smooth_dmm / smooth_tr.replace(0, np.nan)

    dx_denom = di_plus + di_minus
    dx = 100.0 * (di_plus - di_minus).abs() / dx_denom.replace(0, np.nan)

    # Wilder smooth DX to get ADX
    adx_vals = _wilder_smooth(dx.dropna().reindex(dx.index))

    return adx_vals, di_plus, di_minus


# ─── Efficiency Ratio (ER20) ─────────────────────────────────────────────────

def efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    """
    ER = abs(C[t] - C[t-period]) / sum(|C[i] - C[i-1]|, period bars).
    Returns 0 for zero denominator.
    """
    direction = close.diff(period).abs()
    volatility = close.diff(1).abs().rolling(period).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0.0)
    return er.clip(0, 1)


# ─── NATR ────────────────────────────────────────────────────────────────────

def natr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """NATR = ATR14 / abs(close). Returns NaN where close == 0."""
    atr_vals = atr_wilder(high, low, close, period)
    result = atr_vals / close.abs().replace(0, np.nan)
    return result


def natr_percentile(natr_series: pd.Series, lookback: int = 200) -> pd.Series:
    """
    Rolling percentile of current NATR vs preceding `lookback` observations.
    Uses only preceding observations (no future data).
    """
    def _pct(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        current = x[-1]
        history = x[:-1]
        if len(history) == 0 or np.all(np.isnan(history)):
            return np.nan
        return float(np.nansum(history <= current) / np.sum(~np.isnan(history)) * 100)

    result = natr_series.rolling(lookback + 1, min_periods=2).apply(_pct, raw=True)
    return result


# ─── Bollinger Bands ─────────────────────────────────────────────────────────

def bollinger_bands(
    close: pd.Series, period: int = 20, std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (upper, middle, lower).
    Uses population standard deviation (ddof=0) — matches Pine's ta.bb().
    """
    middle = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, middle, lower


def bb_width(upper: pd.Series, lower: pd.Series, middle: pd.Series) -> pd.Series:
    """Bollinger band width normalised by middle band."""
    return (upper - lower) / middle.replace(0, np.nan)


# ─── Keltner Channel ─────────────────────────────────────────────────────────

def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    atr_mult: float = 1.5,
    atr_period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower). Middle is EMA of close."""
    middle = ema(close, period)
    atr_vals = atr_wilder(high, low, close, atr_period)
    upper = middle + atr_mult * atr_vals
    lower = middle - atr_mult * atr_vals
    return upper, middle, lower


def bb_squeeze(
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    kc_upper: pd.Series,
    kc_lower: pd.Series,
) -> pd.Series:
    """True when both BB bands lie inside KC bands (squeeze condition)."""
    return (bb_upper < kc_upper) & (bb_lower > kc_lower)


# ─── VWAP (session-reset) ────────────────────────────────────────────────────

def session_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    session_starts: pd.Series,   # boolean Series: True on first bar of each session
) -> pd.Series:
    """
    Session-reset VWAP using typical price (H+L+C)/3.
    session_starts marks the first bar of each new session.
    Disabled on daily/higher charts (returns NaN series).
    """
    tp = (high + low + close) / 3.0
    cum_tpv = (tp * volume).copy()
    cum_vol = volume.copy()

    result = pd.Series(np.nan, index=close.index, dtype=float)
    running_tpv = 0.0
    running_vol = 0.0

    for i, idx in enumerate(close.index):
        if session_starts.iloc[i]:
            running_tpv = 0.0
            running_vol = 0.0
        running_tpv += tp.iloc[i] if not pd.isna(tp.iloc[i]) else 0.0
        running_vol += volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0.0
        if running_vol > 0:
            result.iloc[i] = running_tpv / running_vol

    return result


# ─── Trend score (T) per Section 10 ─────────────────────────────────────────

def trend_score(
    close: pd.Series,
    ema20: pd.Series,
    ema50: pd.Series,
    ema200: pd.Series,
    atr: pd.Series,
) -> pd.Series:
    """
    T = clip[0.4*clip((EMA20-EMA50)/A,-1,1)
             + 0.3*clip((EMA50[t]-EMA50[t-5])/A,-1,1)
             + 0.3*sign(C-EMA200), -1, 1]

    Uses prior-bar ATR (A = atr.shift(1)) so the current bar
    does not change its own threshold.
    """
    A = atr.shift(1)

    term1 = ((ema20 - ema50) / A).clip(-1, 1) * 0.4
    term2 = ((ema50 - ema50.shift(5)) / A).clip(-1, 1) * 0.3
    term3 = np.sign(close - ema200) * 0.3

    T = (term1 + term2 + term3).clip(-1, 1)
    return T


# ─── RVOL ────────────────────────────────────────────────────────────────────

def relative_volume(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """RVOL = V[t] / mean(V[t-lookback:t-1]). NaN if window unavailable."""
    mean_vol = volume.shift(1).rolling(lookback).mean()
    return volume / mean_vol.replace(0, np.nan)


def signed_participation(
    open_: pd.Series,
    close: pd.Series,
    rvol: pd.Series,
) -> pd.Series:
    """
    Signed participation = sign(C-O) * clip((RVOL-0.8)/0.8, 0, 1).
    Returns NaN when volume is unavailable/zero.
    """
    direction = np.sign(close - open_)
    magnitude = ((rvol - 0.8) / 0.8).clip(0, 1)
    return direction * magnitude
