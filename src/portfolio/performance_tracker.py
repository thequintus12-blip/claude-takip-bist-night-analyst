"""
Haftalık portföy döngüsünün GERÇEK (simüle edilmiş, komisyon/kaymasız)
getirisini izler.

Neden gerekli: GA'nın out-of-sample fitness'ı (genetic_optimizer.py) ve
agent ağırlıkları (feedback_loop.py) hep SENTETİK, tek-tek sinyal bazlı bir
backtest üzerinden hesaplanır ("bu sinyalden N gün sonra ne oldu"). Ama
kullanıcının fiilen takip ettiği şey budur DEĞİL — kullanıcı her hafta
sistemin seçtiği hisse SEPETİNİ Pazartesi'den Cuma'ya kadar tutuyor
(bkz. cycle_manager.py). Bu iki şey aynı sonucu vermeyebilir (örneğin
hafta ortasında bir hisse "kötüleşip" biri onunla değiştirilmemişse, o
hisse Cuma'ya kadar sepette kalır — sentetik backtest bunu hiç modellemez).

Bu modül, cycle_manager.py her Cuma pozisyonları "kapatırken" çağrılır ve
her tamamlanan pozisyonun (giriş fiyatı -> çıkış fiyatı) gerçekleşen
getirisini data/processed/cycle_performance_log.csv'ye ekler. Bu, sistemin
kendini geliştirip geliştirmediğinin TEK somut, denetlenebilir kanıtıdır.
"""

from __future__ import annotations

import pandas as pd

from config import settings

COLUMNS = [
    "entered_on", "exited_on", "ticker", "entry_price", "exit_price",
    "realized_return", "holding_days", "exit_reason",
]


def log_completed_positions(rows: list[dict]) -> None:
    """Tamamlanan pozisyonları (bkz. COLUMNS) log dosyasına ekler. `rows`
    boşsa hiçbir şey yapmaz. entry_price/exit_price bilinmiyorsa
    realized_return alanı None olarak kaydedilir (satır yine de eklenir —
    veri eksikliği sessizce gizlenmez, denetlenebilir kalır)."""
    if not rows:
        return
    settings.CYCLE_PERFORMANCE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows, columns=COLUMNS)
    if settings.CYCLE_PERFORMANCE_LOG_FILE.exists():
        existing = pd.read_csv(settings.CYCLE_PERFORMANCE_LOG_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(settings.CYCLE_PERFORMANCE_LOG_FILE, index=False)


def load_performance_log() -> pd.DataFrame:
    if not settings.CYCLE_PERFORMANCE_LOG_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(settings.CYCLE_PERFORMANCE_LOG_FILE, parse_dates=["entered_on", "exited_on"])


def summarize_performance(df: pd.DataFrame | None = None) -> dict:
    """Kümülatif ve son ~8 haftalık (56 takvim günü) özet istatistikler.
    Satır sayısına değil TARİHE göre pencereleme yapılır — böylece haftada
    kaç pozisyon tutulduğundan bağımsız, doğru bir "son dönem" tanımı
    olur."""
    if df is None:
        df = load_performance_log()
    valid = df.dropna(subset=["realized_return"])
    if valid.empty:
        return {
            "n_positions": 0, "win_rate": float("nan"), "avg_return": float("nan"),
            "cumulative_return": float("nan"), "recent_avg_return": float("nan"),
        }
    cumulative = (1 + valid["realized_return"]).prod() - 1
    cutoff = valid["exited_on"].max() - pd.Timedelta(days=56)
    recent = valid[valid["exited_on"] >= cutoff]
    return {
        "n_positions": int(len(valid)),
        "win_rate": round(float((valid["realized_return"] > 0).mean()), 4),
        "avg_return": round(float(valid["realized_return"].mean()), 4),
        "cumulative_return": round(float(cumulative), 4),
        "recent_avg_return": round(float(recent["realized_return"].mean()), 4) if not recent.empty else float("nan"),
    }
