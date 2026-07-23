from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_collector.preprocessor import MIN_REQUIRED_ROWS, clean_ohlcv


def _valid_df(n: int = 250) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    close = np.maximum(close, 1)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": np.full(n, 1_000_000.0),
    }, index=dates)


def test_clean_ohlcv_rejects_insufficient_rows():
    df = _valid_df(n=50)
    result = clean_ohlcv(df, "TEST")
    assert result is None


def test_clean_ohlcv_accepts_sufficient_valid_data():
    df = _valid_df(n=MIN_REQUIRED_ROWS + 10)
    result = clean_ohlcv(df, "TEST")
    assert result is not None
    assert len(result) >= MIN_REQUIRED_ROWS


def test_clean_ohlcv_drops_inconsistent_rows():
    df = _valid_df(n=MIN_REQUIRED_ROWS + 10)
    # Bozuk bir satır ekle: Low > High (imkansız)
    df.loc[df.index[5], "Low"] = df.loc[df.index[5], "High"] + 10
    result = clean_ohlcv(df, "TEST")
    assert result is not None
    assert df.index[5] not in result.index


def test_clean_ohlcv_handles_none_and_empty():
    assert clean_ohlcv(None, "TEST") is None
    assert clean_ohlcv(pd.DataFrame(), "TEST") is None
