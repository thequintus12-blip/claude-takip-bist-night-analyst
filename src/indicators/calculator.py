"""
Teknik indikatörler — TA-Lib bağımlılığı OLMADAN, sıfırdan vektörize
implementasyonlar (Streamlit Cloud uyumluluğu için).

Look-ahead notu: Burada kullanılan rolling()/ewm() pencereleri doğası gereği
sadece geçmiş + an itibarıyla mevcut satırı kullanır (pandas varsayılan
davranışı, "backward-looking"). Bu fonksiyonların KENDİSİ look-ahead riski
taşımaz. Risk, bu indikatörlerin *hangi tarihe kadar olan veri ile*
çağrıldığında ortaya çıkar — bu sorumluluğu çağıran taraf
(src/agents/base_agent.py:get_data_as_of) üstlenir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder'ın orijinal RSI formülü (ewm alpha=1/period ile smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    return rsi_values


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
          signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Wilder'ın ATR'si."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.DataFrame:
    """Wilder'ın ADX'i (+DI, -DI, ADX)."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    atr_smooth = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smooth)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_values = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_values})


def bollinger_bands(close: pd.Series, period: int = 20,
                     num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=1)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width})


def bb_squeeze_percentile(bb_width: pd.Series, lookback: int = 60) -> pd.Series:
    """Mevcut BB genişliğinin son `lookback` gün içindeki percentile sırası.
    Düşük percentile (örn. <%10) = sıkışma (breakout öncesi), yüksek değer
    bile olsa, açılmış volatilite sonrası da doğru tespit edilebilmesi için
    'son N günün minimum genişliği' yaklaşımı yerine percentile rank kullanılır."""
    return bb_width.rolling(window=lookback, min_periods=lookback).apply(
        lambda window: pd.Series(window).rank(pct=True).iloc[-1], raw=False
    )


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    avg_volume = volume.shift(1).rolling(window=period, min_periods=period).mean()
    return volume / avg_volume.replace(0, np.nan)


def relative_strength_vs_benchmark(close: pd.Series, benchmark_close: pd.Series,
                                    period: int = 20) -> pd.Series:
    """Hissenin son `period` günlük getirisi - benchmark'ın aynı dönem getirisi."""
    stock_ret = close / close.shift(period) - 1
    bench_ret = benchmark_close / benchmark_close.shift(period) - 1
    bench_ret = bench_ret.reindex(stock_ret.index, method="ffill")
    return stock_ret - bench_ret


def resistance_breakout(high: pd.Series, close: pd.Series,
                         lookback: int = 20) -> pd.Series:
    """Önceki `lookback` günün (bugünü hariç tutarak) en yüksek seviyesinin
    üzerine kapanış yapıldı mı? shift(1) ile bugünün kendi mumu pencereden
    dışlanır — aksi halde bugünün yükseği kendi kırılımını maskeler."""
    prior_high = high.shift(1).rolling(window=lookback, min_periods=lookback).max()
    return close > prior_high


def candle_quality(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series) -> pd.Series:
    """Basit mum kalitesi skoru: gövde/aralık oranı (0-1). Yüksek değer =
    güçlü yönlü mum (uzun fitilsiz, kararlı kapanış)."""
    rng = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    return (body / rng).clip(0, 1)


def is_green(open_: pd.Series, close: pd.Series) -> pd.Series:
    return close > open_
