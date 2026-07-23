from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.indicators import calculator as ind


def _make_close_series(n: int = 100, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.015, n)
    close = 100 * np.cumprod(1 + returns)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(close, index=dates)


def test_rsi_bounds():
    close = _make_close_series()
    rsi = ind.rsi(close, period=14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_constant_uptrend_is_high():
    dates = pd.bdate_range("2024-01-01", periods=30)
    close = pd.Series(np.linspace(100, 130, 30), index=dates)
    rsi = ind.rsi(close, period=14)
    assert rsi.dropna().iloc[-1] > 90


def test_macd_columns_present():
    close = _make_close_series()
    macd_df = ind.macd(close)
    assert set(["macd", "signal", "histogram"]).issubset(macd_df.columns)


def test_atr_non_negative():
    close = _make_close_series()
    high = close * 1.01
    low = close * 0.99
    atr = ind.atr(high, low, close, period=14)
    assert (atr.dropna() >= 0).all()


def test_resistance_breakout_excludes_current_bar_from_its_own_window():
    """Bugünün yükseği, bugünün kendi kırılım kontrolünü etkilememeli
    (shift(1) doğrulaması)."""
    dates = pd.bdate_range("2024-01-01", periods=25)
    high = pd.Series([100] * 24 + [200], index=dates)  # son gün ani sıçrama
    close = pd.Series([99] * 24 + [199], index=dates)
    breakout = ind.resistance_breakout(high, close, lookback=20)
    assert breakout.iloc[-1] == True  # noqa: E712 -- önceki 20 günün maksimumu 100, bugün 199 > 100


def test_relative_volume_uses_prior_average_not_current_day():
    volume = pd.Series([100] * 20 + [1000], index=pd.bdate_range("2024-01-01", periods=21))
    rel_vol = ind.relative_volume(volume, period=20)
    # Son günün rel_volume'u, son günü İÇERMEYEN bir ortalamaya göre hesaplanmalı
    # (shift(1) kullanıldığı için) -> 1000 / 100 = 10
    assert rel_vol.iloc[-1] == pytest.approx(10.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
