"""
Backtest sonuçlarından performans metrikleri (TA-Lib/vectorbt'siz, sadece
pandas/numpy ile).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def win_rate(df: pd.DataFrame) -> float:
    """Sadece pozisyon açılan (position != 0) ve sonucu bilinen satırlar
    üzerinden kazanma oranı."""
    active = df[(df["position"] != 0) & df["result_known"]]
    if active.empty:
        return float("nan")
    wins = (active["strategy_return"] > 0).sum()
    return round(wins / len(active), 4)


def average_return(df: pd.DataFrame) -> float:
    active = df[(df["position"] != 0) & df["result_known"]]
    if active.empty:
        return float("nan")
    return round(active["strategy_return"].mean(), 4)


def return_std(df: pd.DataFrame) -> float:
    """İşlemler arası getiri dağılımının standart sapması (avg_return ile
    AYNI birimde — yani doğrudan karşılaştırılabilir/toplanabilir, Sharpe
    gibi yıllıklandırılmış/normalize edilmiş bir oran DEĞİL). GA'nın
    fitness fonksiyonunda "getiriyi bozmadan volatiliteyi cezalandırmak"
    için kullanılır (bkz. genetic_optimizer.py::_robust_fitness_score)."""
    active = df[(df["position"] != 0) & df["result_known"]]
    if len(active) < 2:
        return float("nan")
    return round(active["strategy_return"].std(ddof=1), 4)


def sharpe_ratio(df: pd.DataFrame, periods_per_year: int = 52) -> float:
    """Basitleştirilmiş Sharpe (risksiz oran=0 varsayımıyla), haftalık
    swing trading ufkuna göre yıllıklandırılmış."""
    active = df[(df["position"] != 0) & df["result_known"]]
    if len(active) < 2 or active["strategy_return"].std(ddof=1) == 0:
        return float("nan")
    mean_ret = active["strategy_return"].mean()
    std_ret = active["strategy_return"].std(ddof=1)
    return round((mean_ret / std_ret) * np.sqrt(periods_per_year), 4)


def max_drawdown(df: pd.DataFrame) -> float:
    active = df[df["result_known"]].copy()
    if active.empty:
        return float("nan")
    cumulative = (1 + active["strategy_return"].fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return round(drawdown.min(), 4)


def trade_count(df: pd.DataFrame) -> int:
    return int((df["position"] != 0).sum())


def summarize(df: pd.DataFrame) -> dict:
    return {
        "trade_count": trade_count(df),
        "win_rate": win_rate(df),
        "avg_return": average_return(df),
        "return_std": return_std(df),
        "sharpe": sharpe_ratio(df),
        "max_drawdown": max_drawdown(df),
    }
