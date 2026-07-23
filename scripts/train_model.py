"""
Periyodik parametre optimizasyon scripti. GitHub Actions'da haftalık bir
ayrı job olarak çalıştırılması önerilir (bkz.
.github/workflows/nightly_analysis.yml içindeki weekly job).

run_daily_analysis.py'den BAĞIMSIZDIR — günlük sinyal üretimini bloklamaz,
sadece data/models/agent_params.json dosyasını günceller; bir sonraki gece
taraması bu güncel parametreleri otomatik olarak okur.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.symbols import get_benchmark_ticker
from src.data_collector.collector import fetch_watchlist_data
from src.data_collector.preprocessor import clean_watchlist_data
from src.learning.genetic_optimizer import optimize_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_model")


def main() -> None:
    logger.info("=== Genetik Parametre Optimizasyonu Başladı ===")

    raw_data = fetch_watchlist_data(save_raw=False)
    raw_data = clean_watchlist_data(raw_data)
    if not raw_data:
        logger.error("Veri yok, optimizasyon durduruldu.")
        return

    benchmark_ticker = get_benchmark_ticker()
    result = optimize_parameters(raw_data, benchmark_ticker)

    logger.info("Optimizasyon tamamlandı.")
    logger.info("Train fitness: %.4f", result.get("train_fitness", float("nan")))
    logger.info("Bu haftanın adayı — out-of-sample performans: %.4f", result.get("out_of_sample_performance", float("nan")))
    if result.get("current_oos_performance") is not None:
        logger.info("Canlıdaki mevcut parametreler — out-of-sample performans: %.4f", result["current_oos_performance"])
    if result.get("adopted"):
        logger.info("✅ BENİMSENDİ: yeni parametreler kanıtlanmış bir iyileşme sağladığı için canlıya alındı.")
    else:
        logger.info(
            "⛔ REDDEDİLDİ: bu haftanın adayı canlıdaki parametrelerden daha iyi değildi, "
            "agent_params.json değiştirilmedi (sistem geriye gitmez, sadece kanıtlanmış "
            "iyileşmelerde ilerler)."
        )
    logger.info(
        "Not: out-of-sample değeri, GA sürecinin hiç görmediği test "
        "pencerelerinde ölçülmüştür — gerçek genelleme performansının "
        "göstergesidir."
    )


if __name__ == "__main__":
    main()
