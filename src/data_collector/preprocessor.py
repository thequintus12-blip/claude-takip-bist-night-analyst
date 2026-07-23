"""
Ham OHLCV verisini temizler ve agent/indikatör hesaplamaları için hazır hale
getirir. Bu modül veriyi *kesmez/sınırlamaz* — sadece kaliteyi garanti eder.
Tarihsel sınırlama (look-ahead önleme) src/agents/base_agent.py içindeki
get_data_as_of() ile yapılır; bu modülde DEĞİL.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

MIN_REQUIRED_ROWS = 220  # MA200 gibi en uzun pencereyi hesaplayabilmek için


def clean_ohlcv(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame | None:
    """Eksik/bozuk satırları temizler, minimum veri uzunluğunu doğrular."""
    if df is None or df.empty:
        return None

    df = df.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Negatif veya sıfır fiyat/hacim gibi bozuk satırları at
    price_cols = ["Open", "High", "Low", "Close"]
    df = df.dropna(subset=price_cols)
    df = df[(df[price_cols] > 0).all(axis=1)]
    df["Volume"] = df["Volume"].fillna(0)
    df = df[df["Volume"] >= 0]

    # OHLC tutarlılığı: High >= Low, High >= Close/Open, Low <= Close/Open
    consistent = (
        (df["High"] >= df["Low"])
        & (df["High"] >= df["Close"])
        & (df["High"] >= df["Open"])
        & (df["Low"] <= df["Close"])
        & (df["Low"] <= df["Open"])
    )
    n_bad = (~consistent).sum()
    if n_bad > 0:
        logger.warning("%s: %d tutarsız OHLC satırı atıldı.", ticker, n_bad)
    df = df[consistent]

    if len(df) < MIN_REQUIRED_ROWS:
        logger.warning(
            "%s: yetersiz veri (%d satır < %d minimum), elendi.",
            ticker, len(df), MIN_REQUIRED_ROWS,
        )
        return None

    return df


def clean_watchlist_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Tüm watchlist için temizleme uygular, geçersiz olanları eler."""
    cleaned = {}
    for ticker, df in data.items():
        result = clean_ohlcv(df, ticker)
        if result is not None:
            cleaned[ticker] = result
    logger.info("Temizlik sonrası %d/%d ticker geçerli.", len(cleaned), len(data))
    return cleaned
