"""
Ana gece çalıştırma scripti. GitHub Actions tarafından piyasa kapanışı
sonrası tetiklenir (bkz. .github/workflows/nightly_analysis.yml).

Adımlar:
  1. Veri çek (yfinance) + temizle
  2. Feature'ları hesapla (tüm geçmiş veri ile, ısınma payı dahil)
  3. Bugüne (as_of_date = son işlem günü) kadar olan veriyle sinyal üret
  4. Sonuçları data/processed/latest_signals.csv'ye yaz
  5. Bugünün tahminlerini predictions_log.csv'ye ekle (result henüz bilinmez)
  6. Daha önce üretilmiş ve sonucu artık bilinen tahminleri değerlendir
  7. Agent ağırlıklarını feedback'e göre güncelle
  8. Haftalık portföy döngüsünü ilerlet (Pazartesi seçim, Salı-Perşembe
     gözden geçirme/değiştirme önerisi, Cuma satış + gelecek hafta adayları)
  9. (Opsiyonel) Telegram bildirimi gönder

Bu script, tüm veriyi as_of_date'e kadar keserek çalışır — "bugünün
taramasını yaparken yarının verisini görme" riski mimari olarak
(get_data_as_of + simulate_signals ayrımı) zaten engellenmiştir.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import settings
from config.symbols import get_benchmark_ticker, get_yfinance_watchlist
from src.agents.supervisor import analyze_watchlist, apply_confirmation_gate
from src.backtest.engine import EVALUATION_HORIZON_DAYS
from src.data_collector.collector import fetch_watchlist_data
from src.data_collector.preprocessor import clean_watchlist_data
from src.indicators.feature_engineer import build_features, build_features_for_watchlist
from src.learning.feedback_loop import (
    append_predictions, evaluate_due_predictions, load_weights,
    update_weights_from_feedback,
)
from src.portfolio.cycle_manager import build_weekly_cycle_report, cycle_summary_text
from src.reporter.notifier import notify_summary
from src.reporter.report_generator import results_to_dataframe, save_full_results, summary_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(settings.LOGS_DIR / "analyst.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("run_daily_analysis")


def main() -> None:
    logger.info("=== BIST Gece Analizi Başladı ===")

    # 1. Veri çek + temizle
    raw_data = fetch_watchlist_data(save_raw=True)
    raw_data = clean_watchlist_data(raw_data)
    if not raw_data:
        logger.error("Hiçbir hisse için geçerli veri yok, çalışma durduruldu.")
        return

    benchmark_ticker = get_benchmark_ticker()

    # 2. Feature'ları hesapla
    params = {}  # agent_params.json varsa supervisor/feature_engineer içinde okunabilir
    if settings.PARAMS_FILE.exists():
        import json
        with open(settings.PARAMS_FILE, "r", encoding="utf-8") as f:
            params = json.load(f).get("params", {})
    # NOT (düzeltilen hata): GA'nın bağımsız örneklediği "min_risk_reward"
    # ve ATR çarpanları, kayıtlı dosyada ulaşılması imkansız bir eşik
    # oluşturmuş olabilir (bkz. config/settings.py::sanitize_params
    # docstring'i). Bu satır olmadan, böyle bozuk bir kombinasyon
    # kaydedildiği sürece hiçbir AL sinyali asla onaylanamaz.
    params = settings.sanitize_params(params)

    features_by_ticker = build_features_for_watchlist(raw_data, benchmark_ticker, params)
    benchmark_features = None
    if benchmark_ticker in raw_data:
        benchmark_features = build_features(raw_data[benchmark_ticker], None, params)

    # 3. as_of_date = en son ortak işlem günü
    as_of_date = min(df.index.max() for df in features_by_ticker.values())
    logger.info("Analiz tarihi (as_of_date): %s", as_of_date.date())

    # 4. Sinyal üret
    weights = load_weights()
    results = analyze_watchlist(features_by_ticker, as_of_date, weights, params, benchmark_features)

    # 4b. İkinci göz doğrulaması: sadece AL sinyallerine uygulanır
    results = apply_confirmation_gate(results, features_by_ticker, as_of_date, params)

    # 5. Sonuçları kaydet (özet CSV + agent bazlı tam detay JSON)
    report_df = results_to_dataframe(results)
    report_df.to_csv(settings.LATEST_SIGNALS_FILE, index=False)
    save_full_results(results, settings.LATEST_SIGNALS_DETAIL_FILE)
    logger.info(summary_text(results))

    # 6. Tahminleri log'a ekle (sonucu henüz bilinmiyor)
    # NOT (düzeltilen hata): "as_of_date + N takvim günü" hafta sonuna denk
    # geldiğinde gerçek ufuk 5 İŞLEM günü değil, 3-4 işlem gününe düşüyordu
    # (backtest/GA ise her zaman tam 5 İŞLEM günü/satır kaydırıyor — bkz.
    # backtest/engine.py::compute_forward_returns). Takvim günü yerine iş
    # günü (BDay) kullanmak, canlı feedback döngüsünün ölçtüğü ufku GA'nın
    # optimize ettiği ufukla tutarlı hale getirir (resmi tatiller hâlâ
    # hesaba katılmıyor, ama hafta sonu kayması artık yok).
    new_rows = []
    result_date = as_of_date + pd.tseries.offsets.BDay(EVALUATION_HORIZON_DAYS)
    for r in results:
        for sig in r.get("agent_signals", []):
            new_rows.append({
                "as_of_date": as_of_date, "ticker": r["ticker"], "agent": sig["agent"],
                "signal": sig["signal"], "signal_value": sig["signal_value"],
                "confidence": sig["confidence"], "weight": sig["weight"],
                "result_available_date": result_date,
                "actual_return": None, "was_correct": None, "evaluated": False,
            })
    if new_rows:
        append_predictions(pd.DataFrame(new_rows))

    # 7. Sonucu artık bilinen geçmiş tahminleri değerlendir
    price_lookup = {ticker: df["Close"] for ticker, df in raw_data.items()}
    evaluate_due_predictions(price_lookup, as_of_date)

    # 8. Ağırlıkları güncelle
    new_weights = update_weights_from_feedback()
    logger.info("Güncel agent ağırlıkları: %s", new_weights)

    # 8b. Haftalık portföy döngüsünü ilerlet
    cycle_report = build_weekly_cycle_report(results, as_of_date)
    cycle_text = cycle_summary_text(cycle_report)
    settings.PORTFOLIO_CYCLE_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(settings.PORTFOLIO_CYCLE_REPORT_FILE, "w", encoding="utf-8") as f:
        _json.dump(cycle_report, f, ensure_ascii=False, indent=2)
    logger.info(cycle_text)

    # 9. Bildirim (opsiyonel)
    top_buys = report_df[report_df["Sinyal"] == "AL"]["Hisse"].head(5).tolist()
    notify_summary(summary_text(results), top_buys, cycle_summary=cycle_text)

    logger.info("=== BIST Gece Analizi Tamamlandı ===")


if __name__ == "__main__":
    main()
