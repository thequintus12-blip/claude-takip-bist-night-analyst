"""
Look-ahead güvenli backtest motoru.

Temel disiplin:
  sinyal[t]  = f(veri[0:t])           -> sadece geçmiş + an itibarıyla bilinen
  getiri[t]  = fiyat[t+N] / fiyat[t] - 1   -> N gün sonraki, henüz bilinmeyen sonuç

Bu iki hesaplama asla aynı veri penceresini paylaşmaz. walk_forward_splits()
fonksiyonu, genetik optimizasyon gibi parametre arama süreçlerinin de
"train" penceresinde optimize edip "test" penceresinde asla görmediği veriyle
değerlendirilmesini sağlar.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import EVALUATION_HORIZON_DAYS


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(dates: pd.DatetimeIndex, n_windows: int = 4,
                         train_ratio: float = 0.6) -> list[WalkForwardWindow]:
    """Tarih index'ini, train penceresi her zaman test penceresinden ÖNCE
    biten, birbirini izleyen (rolling) n_windows parçaya böler.

    Önemli: bir pencerenin test bölgesi, bir sonraki pencerenin train
    bölgesine sızdırılmaz çünkü her pencere bağımsız ve ardışık tarih
    aralıklarına dayanır (overlap yok).
    """
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    total = len(dates)
    if total < n_windows * 20:
        raise ValueError("Walk-forward için yeterli veri yok (en az ~20 gün/pencere gerekir).")

    chunk_size = total // n_windows
    windows = []
    for i in range(n_windows):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_windows - 1 else total
        chunk = dates[start_idx:end_idx]
        if len(chunk) < 10:
            continue
        split_idx = max(1, int(len(chunk) * train_ratio))
        train = chunk[:split_idx]
        test = chunk[split_idx:]
        if len(test) == 0:
            continue
        windows.append(WalkForwardWindow(
            train_start=train[0], train_end=train[-1],
            test_start=test[0], test_end=test[-1],
        ))
    return windows


def compute_forward_returns(close: pd.Series, horizon: int = None) -> pd.Series:
    """Gün t için, t+horizon gününe kadar olan getiri. Bu seri SADECE
    sonuç değerlendirmesinde kullanılır, asla sinyal üretiminde değil."""
    horizon = horizon or EVALUATION_HORIZON_DAYS
    return close.shift(-horizon) / close - 1


def simulate_signals(
    signals: pd.Series, close: pd.Series, horizon: int = None,
) -> pd.DataFrame:
    """Verilen sinyal serisi (-1/0/1 veya AL/SAT/BEKLE) için forward-return
    bazlı basit simülasyon. signals[t] zaten t gününe kadarki veriyle
    üretilmiş olmalı (çağıran taraf bunu garanti eder); bu fonksiyon sadece
    sonucu ölçer, sinyal üretmez.
    """
    horizon = horizon or EVALUATION_HORIZON_DAYS
    fwd_returns = compute_forward_returns(close, horizon)

    df = pd.DataFrame({"signal": signals, "close": close, "forward_return": fwd_returns})
    df["position"] = df["signal"].map(
        lambda s: 1 if s in ("AL", 1) else (-1 if s in ("SAT", -1) else 0)
    )
    df["strategy_return"] = df["position"] * df["forward_return"]

    # Sonucu henüz bilinmeyen (forward_return NaN olan, yani son `horizon`
    # gün) satırlar değerlendirmeye katılmaz.
    df["result_known"] = df["forward_return"].notna()

    return df
