"""src/learning/genetic_optimizer.py için testler: (1) fitness fonksiyonunun
getiri volatilitesini cezalandırdığını, (2) "ratchet" korumasının canlıdaki
parametrelerden daha iyi olmayan bir adayı ASLA benimsemediğini, daha iyi
olanı ise benimsediğini doğrular.

Gerçek market verisiyle tüm GA'yı (20 birey × 14 jenerasyon × 3 pencere ×
50 hisse) çalıştırmak yavaş ve deterministik olmayacağından,
`_fitness_on_precomputed_features` ve `_build_features_for_individual`
sahte (mock) fonksiyonlarla değiştirilir — bu sayede ratchet mantığı,
GA'nın kendi arama sürecinden bağımsız olarak, hızlı ve deterministik
şekilde test edilir.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.learning import genetic_optimizer as go


def _make_synthetic_ohlcv(n_days: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_days))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
         "Volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float)},
        index=dates,
    )


@pytest.fixture(autouse=True)
def _isolate_params_files(tmp_path, monkeypatch):
    monkeypatch.setattr(go, "PARAMS_FILE", tmp_path / "agent_params.json")
    monkeypatch.setattr(go, "PARAMS_HISTORY_FILE", tmp_path / "params_history.csv")
    yield


def test_fitness_penalizes_return_volatility_at_equal_average_return():
    steady = {"avg_return": 0.02, "trade_count": 5, "max_drawdown": -0.01, "return_std": 0.005}
    choppy = {"avg_return": 0.02, "trade_count": 5, "max_drawdown": -0.01, "return_std": 0.05}

    steady_score = go._robust_fitness_score(steady)
    choppy_score = go._robust_fitness_score(choppy)

    assert steady_score > choppy_score


def test_fitness_ignores_missing_return_std_gracefully():
    metrics = {"avg_return": 0.02, "trade_count": 5, "max_drawdown": -0.01, "return_std": float("nan")}
    reliability = min(1.0, 5 / go.MIN_RELIABLE_TRADES)
    expected = (0.02 - 0.3 * 0.01) * reliability
    assert go._robust_fitness_score(metrics) == pytest.approx(expected)


def test_first_run_always_adopts_ga_result(monkeypatch):
    """Kayıtlı hiçbir parametre yokken (ilk çalıştırma), GA'nın bulduğu
    sonuç -inf'e karşı karşılaştırıldığı için her zaman benimsenmelidir."""
    monkeypatch.setattr(go, "_build_features_for_individual", lambda raw_data, benchmark_ticker, params: ({}, None))
    monkeypatch.setattr(
        go, "_fitness_on_precomputed_features",
        lambda features_by_ticker, benchmark_features, params, window_start, window_end: 0.05,
    )

    raw_data = {"XXX.IS": _make_synthetic_ohlcv(), "XU100.IS": _make_synthetic_ohlcv(seed=2)}
    result = go.optimize_parameters(
        raw_data, "XU100.IS", population_size=4, generations=2, n_windows=2, ticker_sample_size=10,
    )

    assert result["adopted"] is True
    assert go.PARAMS_FILE.exists()
    with open(go.PARAMS_FILE, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["adopted"] is True


def test_ratchet_rejects_candidate_that_does_not_beat_current_params(monkeypatch):
    """Canlıdaki parametreler zaten mümkün olan en iyi (sahte) fitness'ı
    alıyorsa, GA'nın ürettiği HİÇBİR rastgele birey onu geçemez — dosya
    değişmemeli."""
    current = dict(go.DEFAULT_PARAMS)
    current["rsi_period"] = 15
    with open(go.PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump({"params": current}, f)
    original_mtime = go.PARAMS_FILE.stat().st_mtime

    def fake_fitness(features_by_ticker, benchmark_features, params, window_start, window_end):
        # current_params'ın kendi rsi_period'u tam olarak en yüksek skoru
        # (0.0) alır; başka HİÇBİR rsi_period değeri bunu geçemez.
        return -abs(params.get("rsi_period", 15) - 15) / 100.0

    monkeypatch.setattr(go, "_build_features_for_individual", lambda raw_data, benchmark_ticker, params: ({}, None))
    monkeypatch.setattr(go, "_fitness_on_precomputed_features", fake_fitness)

    raw_data = {"XXX.IS": _make_synthetic_ohlcv(), "XU100.IS": _make_synthetic_ohlcv(seed=2)}
    result = go.optimize_parameters(
        raw_data, "XU100.IS", population_size=6, generations=3, n_windows=2, ticker_sample_size=10,
    )

    assert result["adopted"] is False
    assert result["params"]["rsi_period"] == 15  # canlıdaki parametreler korunuyor
    assert go.PARAMS_FILE.stat().st_mtime == original_mtime  # dosyaya HİÇ dokunulmadı


def test_ratchet_adopts_candidate_that_beats_current_params(monkeypatch):
    """Canlıdaki parametreler kötüyse (düşük sahte fitness), GA'nın
    bulduğu herhangi bir farklı birey onu geçmeli ve dosya güncellenmeli."""
    current = dict(go.DEFAULT_PARAMS)
    current["rsi_period"] = 15
    with open(go.PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump({"params": current}, f)

    def fake_fitness(features_by_ticker, benchmark_features, params, window_start, window_end):
        # current_params'ın rsi_period'undan UZAKLAŞMAK ödüllendirilir —
        # yani rastgele üretilen hemen hemen her birey ondan daha iyi olur.
        return abs(params.get("rsi_period", 15) - 15) / 100.0

    monkeypatch.setattr(go, "_build_features_for_individual", lambda raw_data, benchmark_ticker, params: ({}, None))
    monkeypatch.setattr(go, "_fitness_on_precomputed_features", fake_fitness)

    raw_data = {"XXX.IS": _make_synthetic_ohlcv(), "XU100.IS": _make_synthetic_ohlcv(seed=2)}
    result = go.optimize_parameters(
        raw_data, "XU100.IS", population_size=10, generations=4, n_windows=2, ticker_sample_size=10,
    )

    assert result["adopted"] is True
    with open(go.PARAMS_FILE, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["params"]["rsi_period"] != 15


def test_params_history_records_rejected_attempts_too(monkeypatch):
    """GA bir iyileşme bulamasa (adopted=False) bile, bu haftanın
    denemesi params_history.csv'ye kaydedilmeli — aksi halde 'GA kaç
    haftada bir gerçekten bir şey buluyor' hiç izlenemez."""
    current = dict(go.DEFAULT_PARAMS)
    current["rsi_period"] = 15
    with open(go.PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump({"params": current}, f)

    def fake_fitness(features_by_ticker, benchmark_features, params, window_start, window_end):
        return -abs(params.get("rsi_period", 15) - 15) / 100.0

    monkeypatch.setattr(go, "_build_features_for_individual", lambda raw_data, benchmark_ticker, params: ({}, None))
    monkeypatch.setattr(go, "_fitness_on_precomputed_features", fake_fitness)

    raw_data = {"XXX.IS": _make_synthetic_ohlcv(), "XU100.IS": _make_synthetic_ohlcv(seed=2)}
    go.optimize_parameters(raw_data, "XU100.IS", population_size=4, generations=2, n_windows=2, ticker_sample_size=10)

    hist = pd.read_csv(go.PARAMS_HISTORY_FILE)
    assert len(hist) == 1
    assert hist.iloc[0]["adopted"] == False  # noqa: E712
