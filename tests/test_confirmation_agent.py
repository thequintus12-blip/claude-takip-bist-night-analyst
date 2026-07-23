"""
İkinci göz doğrulama katmanı (ConfirmationAgent) için testler.

Bu dosya iki şeyi garanti eder:
  1. Üç kriterin (likidite, risk/ödül, aşırı RSI vetosu) her biri doğru
     tetikleniyor.
  2. ConfirmationAgent de tüm sistem gibi look-ahead güvenli — as_of_date
     sabit kaldığı sürece, DataFrame'e gelecek günler eklense bile sonuç
     değişmemeli.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.agents.confirmation_agent import ConfirmationAgent
from src.indicators.feature_engineer import build_features


def _make_healthy_ohlcv(n_days: int = 100, seed: int = 7) -> pd.DataFrame:
    """Gürültülü (gerçekçi) ve makul likiditeli bir seri — tüm kriterleri
    geçmesi beklenir."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    returns = rng.normal(0.003, 0.015, n_days)
    close = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    high, low = close * 1.008, close * 0.992
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(500_000.0, index=dates)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


def test_all_checks_pass_on_healthy_liquid_data():
    df = _make_healthy_ohlcv()
    features = build_features(df)
    agent = ConfirmationAgent()
    result = agent.review("TEST.IS", features, df.index[-1], {"risk_reward": 2.0})
    assert result.confirmed is True
    assert all(result.checks.values())


def test_low_liquidity_is_rejected():
    df = _make_healthy_ohlcv()
    df = df.copy()
    df["Volume"] = 100.0  # ihmal edilebilir işlem hacmi
    features = build_features(df)
    agent = ConfirmationAgent()
    result = agent.review("TEST.IS", features, df.index[-1], {"risk_reward": 2.0})
    assert result.confirmed is False
    assert result.checks["likidite"] is False


def test_low_risk_reward_is_rejected():
    df = _make_healthy_ohlcv()
    features = build_features(df)
    agent = ConfirmationAgent()
    result = agent.review("TEST.IS", features, df.index[-1], {"risk_reward": 1.1})
    assert result.confirmed is False
    assert result.checks["risk_odul"] is False


def test_extreme_rsi_is_vetoed():
    """IEYHO senaryosu: çok dik, kesintisiz bir yükseliş RSI'ı aşırı
    seviyeye taşır ve tek başına vetoya sebep olmalı."""
    n_days = 100
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    close = pd.Series(np.linspace(100, 200, n_days), index=dates)
    high, low = close * 1.005, close * 0.995
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(500_000.0, index=dates)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})
    features = build_features(df)

    assert features["rsi"].iloc[-1] >= 85, "Test senaryosu RSI'ı yeterince yükseltemedi"

    agent = ConfirmationAgent()
    result = agent.review("TEST.IS", features, df.index[-1], {"risk_reward": 2.0})
    assert result.confirmed is False
    assert result.checks["asiri_rsi"] is False


def test_insufficient_data_is_rejected_not_confirmed():
    df = _make_healthy_ohlcv().iloc[:10]
    features = build_features(df)
    agent = ConfirmationAgent()
    result = agent.review("TEST.IS", features, df.index[-1], {"risk_reward": 2.0})
    assert result.confirmed is False


def test_confirmation_agent_is_look_ahead_safe():
    """En kritik test: ConfirmationAgent de as_of_date'ten sonraki veriye
    duyarsız olmalı — diğer tüm agent'larla aynı garanti."""
    df_long = _make_healthy_ohlcv(n_days=300)
    df_short = df_long.iloc[:200].copy()

    features_short = build_features(df_short)
    features_long = build_features(df_long)
    as_of = df_short.index[180]

    agent = ConfirmationAgent()
    supervisor_result = {"risk_reward": 2.0}
    result_short = agent.review("TEST.IS", features_short, as_of, supervisor_result)
    result_long = agent.review("TEST.IS", features_long, as_of, supervisor_result)

    assert result_short.confirmed == result_long.confirmed
    assert result_short.checks == result_long.checks


def test_custom_params_override_defaults():
    """GA'nın bulduğu parametreler gerçekten etkili oluyor mu?"""
    df = _make_healthy_ohlcv()
    features = build_features(df)

    lenient = ConfirmationAgent(params={"min_liquidity_try": 1.0})
    strict = ConfirmationAgent(params={"min_liquidity_try": 1_000_000_000.0})

    result_lenient = lenient.review("TEST.IS", features, df.index[-1], {"risk_reward": 2.0})
    result_strict = strict.review("TEST.IS", features, df.index[-1], {"risk_reward": 2.0})

    assert result_lenient.checks["likidite"] is True
    assert result_strict.checks["likidite"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
