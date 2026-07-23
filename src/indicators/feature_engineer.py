"""
Ham OHLCV verisini, tüm indikatörlerle zenginleştirilmiş bir "feature"
DataFrame'ine dönüştürür. Bu DataFrame her satırda (her gün için) o günün
KAPANIŞINA KADAR bilinen tüm bilgiyi taşır — gelecek hiçbir satıra referans
vermez (bkz. calculator.py'deki shift(1) disiplinleri).
"""

from __future__ import annotations

import pandas as pd

from config.settings import DEFAULT_PARAMS
from src.indicators import calculator as ind


def build_features(df: pd.DataFrame, benchmark_close: pd.Series | None = None,
                    params: dict | None = None) -> pd.DataFrame:
    """Tek bir hisse için tüm indikatörleri hesaplar ve orijinal OHLCV ile
    birlikte tek bir DataFrame'de döndürür.

    Parametreler agent_params.json'dan (genetik optimizer çıktısı) override
    edilebilir; verilmezse config/settings.py'deki varsayılanlar kullanılır.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    out = df.copy()

    out["ma_short"] = ind.sma(out["Close"], p["ma_short"])
    out["ma_long"] = ind.sma(out["Close"], p["ma_long"])
    out["trend_up"] = out["ma_short"] > out["ma_long"]

    out["rsi"] = ind.rsi(out["Close"], p["rsi_period"])

    macd_df = ind.macd(out["Close"], p["macd_fast"], p["macd_slow"], p["macd_signal"])
    out = out.join(macd_df)
    out["macd_bullish_cross"] = (
        (macd_df["macd"] > macd_df["signal"])
        & (macd_df["macd"].shift(1) <= macd_df["signal"].shift(1))
    )
    out["macd_bearish_cross"] = (
        (macd_df["macd"] < macd_df["signal"])
        & (macd_df["macd"].shift(1) >= macd_df["signal"].shift(1))
    )

    out["atr"] = ind.atr(out["High"], out["Low"], out["Close"], p["atr_period"])

    adx_df = ind.adx(out["High"], out["Low"], out["Close"], p["adx_period"])
    out = out.join(adx_df)

    bb_df = ind.bollinger_bands(out["Close"], p["bb_period"], p["bb_std"])
    out = out.join(bb_df)
    out["bb_squeeze_pct"] = ind.bb_squeeze_percentile(out["bb_width"], lookback=60)

    out["rel_volume"] = ind.relative_volume(out["Volume"], p["rel_volume_period"])

    out["resistance_breakout"] = ind.resistance_breakout(out["High"], out["Close"], lookback=20)
    out["candle_quality"] = ind.candle_quality(out["Open"], out["High"], out["Low"], out["Close"])
    out["is_green"] = ind.is_green(out["Open"], out["Close"])

    if benchmark_close is not None:
        out["rel_strength_vs_xu100"] = ind.relative_strength_vs_benchmark(
            out["Close"], benchmark_close, period=20
        )
    else:
        out["rel_strength_vs_xu100"] = pd.NA

    return out


def build_features_for_watchlist(
    data: dict[str, pd.DataFrame], benchmark_ticker: str,
    params: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Tüm watchlist için feature DataFrame'lerini üretir."""
    benchmark_close = None
    if benchmark_ticker in data:
        benchmark_close = data[benchmark_ticker]["Close"]

    features = {}
    for ticker, df in data.items():
        if ticker == benchmark_ticker:
            continue
        features[ticker] = build_features(df, benchmark_close, params)
    return features
