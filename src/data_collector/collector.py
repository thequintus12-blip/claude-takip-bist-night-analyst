"""
BIST hisseleri için yfinance üzerinden OHLCV verisi çeker.

Önemli: Bu modül SADECE ham veri indirir. Hiçbir look-ahead riski taşımaz
çünkü indirilen veri zaten geçmişe ait kapanmış mumlardır. Look-ahead riski
bu veriyi *kullanırken* (özellikle backtest ve agent karar anında) ortaya
çıkar — bkz. src/agents/base_agent.py: get_data_as_of().
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import settings
from config.symbols import get_benchmark_ticker, get_yfinance_watchlist

logger = logging.getLogger(__name__)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance bazen MultiIndex kolon döndürür (özellikle batch indirmede).
    Tek seviyeye indirger."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_single(ticker: str, period: str = None, interval: str = None,
                     max_retries: int = 3) -> pd.DataFrame | None:
    """Tek bir ticker için OHLCV verisi indirir, hata durumunda retry yapar."""
    period = period or settings.HISTORY_PERIOD
    interval = interval or settings.HISTORY_INTERVAL

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if df is None or df.empty:
                logger.warning("Boş veri döndü: %s (deneme %d/%d)", ticker, attempt, max_retries)
                time.sleep(1.5)
                continue
            df = _flatten_columns(df)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hata (%s, deneme %d/%d): %s", ticker, attempt, max_retries, exc)
            time.sleep(1.5)
    logger.error("Veri indirilemedi: %s", ticker)
    return None


def download_batch(tickers: list[str], period: str = None,
                    interval: str = None) -> dict[str, pd.DataFrame]:
    """Birden fazla ticker'ı toplu indirir; başarısız olanlar için tek tek
    fallback dener (önceki projelerdeki desenle aynı)."""
    period = period or settings.HISTORY_PERIOD
    interval = interval or settings.HISTORY_INTERVAL

    result: dict[str, pd.DataFrame] = {}
    try:
        raw = yf.download(
            tickers=tickers, period=period, interval=interval,
            group_by="ticker", auto_adjust=True, threads=True, progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Toplu indirme başarısız, tek tek denenecek: %s", exc)
        raw = None

    for ticker in tickers:
        df = None
        if raw is not None:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw.xs(ticker, axis=1, level=0, drop_level=True)
                elif len(tickers) == 1:
                    df = raw
            except (KeyError, ValueError):
                df = None
            if df is not None:
                df = _flatten_columns(df).dropna(how="all")
                if not df.empty:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
                    df = df[~df.index.duplicated(keep="last")]

        if df is None or df.empty:
            df = download_single(ticker, period, interval)

        if df is not None and not df.empty:
            result[ticker] = df
        else:
            logger.error("Watchlist'ten çıkarıldı (veri yok): %s", ticker)

    return result


def fetch_watchlist_data(save_raw: bool = True) -> dict[str, pd.DataFrame]:
    """Watchlist + benchmark için tüm veriyi çeker, opsiyonel olarak ham
    veriyi data/raw altına kaydeder."""
    tickers = get_yfinance_watchlist() + [get_benchmark_ticker()]
    data = download_batch(tickers)

    if save_raw:
        raw_dir = Path(settings.RAW_DATA_DIR)
        for ticker, df in data.items():
            safe_name = ticker.replace(".", "_")
            df.to_parquet(raw_dir / f"{safe_name}.parquet")

    logger.info("Toplam %d/%d ticker için veri çekildi.", len(data), len(tickers))
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_watchlist_data()
