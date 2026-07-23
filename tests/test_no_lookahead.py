"""
Bu test dosyası, projenin en kritik garantisini doğrular: agent'lar ve
backtest motoru gelecekteki hiçbir veriye erişemez.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.agents.base_agent import get_data_as_of
from src.agents.rsi_agent import RSIAgent
from src.backtest.engine import compute_forward_returns, walk_forward_splits
from src.indicators.feature_engineer import build_features


def _make_synthetic_ohlcv(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    returns = rng.normal(0.0005, 0.02, n_days)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.01, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n_days)))
    open_ = close * (1 + rng.normal(0, 0.005, n_days))
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_get_data_as_of_excludes_future_rows():
    df = _make_synthetic_ohlcv()
    cutoff = df.index[150]
    safe = get_data_as_of(df, cutoff)
    assert safe.index.max() == cutoff
    assert (safe.index <= cutoff).all()
    assert len(safe) == 151  # 0..150 inclusive


def test_agent_signal_does_not_change_when_future_data_appended():
    """En kritik test: bir agent'ın ürettiği sinyal, DataFrame'e gelecekteki
    günler eklense bile (as_of_date sabit kaldığı sürece) DEĞİŞMEMELİDİR.
    Değişiyorsa, agent kodunda bir yerde look-ahead sızıntısı vardır.

    Not: df_short, df_long'un bir ALT-KÜMESİ olarak üretilir (iloc ile
    kırpılır), iki ayrı random çağrısıyla DEĞİL — aksi halde RNG akışındaki
    kayma yüzünden günler arasında veri hizası bozulur ve test yanlış
    pozitif "look-ahead leak" raporlar (gerçek bir sızıntı olmadığı halde)."""
    df_long = _make_synthetic_ohlcv(n_days=300)
    df_short = df_long.iloc[:200].copy()

    features_short = build_features(df_short)
    features_long = build_features(df_long)

    as_of_date = df_short.index[180]
    agent = RSIAgent()

    signal_short = agent.analyze("TEST", features_short, as_of_date)
    signal_long = agent.analyze("TEST", features_long, as_of_date)

    assert signal_short.signal_value == pytest.approx(signal_long.signal_value, abs=1e-9)
    assert signal_short.signal == signal_long.signal


def test_forward_returns_use_future_data_only_for_evaluation():
    df = _make_synthetic_ohlcv()
    fwd = compute_forward_returns(df["Close"], horizon=5)
    # Son 5 gün için forward return hesaplanamaz (gelecek veri yok) -> NaN olmalı
    assert fwd.iloc[-5:].isna().all()
    # Ortadaki bir gün için forward return, manuel hesaplamayla eşleşmeli
    t = 100
    expected = df["Close"].iloc[t + 5] / df["Close"].iloc[t] - 1
    assert fwd.iloc[t] == pytest.approx(expected)


def test_walk_forward_windows_train_precedes_test_and_no_overlap_between_windows():
    df = _make_synthetic_ohlcv(n_days=400)
    windows = walk_forward_splits(df.index, n_windows=4)

    assert len(windows) > 0
    for w in windows:
        assert w.train_end < w.test_start, "Train penceresi test penceresinden önce bitmeli."

    # Pencereler arası örtüşme olmamalı (ardışık olmalı)
    for i in range(len(windows) - 1):
        assert windows[i].test_end < windows[i + 1].train_start


def test_min_required_rows_prevents_premature_signals():
    """Yetersiz veri varken agent BEKLE + confidence=0 dönmeli, asla
    güvenilir bir AL/SAT üretmemeli."""
    df = _make_synthetic_ohlcv(n_days=10)
    features = build_features(df)
    agent = RSIAgent()
    result = agent.analyze("TEST", features, df.index[-1])
    assert result.signal == "BEKLE"
    assert result.confidence == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
