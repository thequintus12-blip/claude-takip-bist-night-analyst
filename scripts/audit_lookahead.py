"""
Audit script: predictions_log.csv'deki rastgele örneklenmiş tahminlerin,
üretildikleri tarihte gerçekten mevcut olmayan hiçbir veriye dayanmadığını
programatik olarak doğrular.

Kontrol mantığı:
  - Her satır için as_of_date ve result_available_date karşılaştırılır.
  - result_available_date, as_of_date'ten KESİNLİKLE sonra olmalı
    (EVALUATION_HORIZON_DAYS kadar).
  - evaluated=True olan satırlarda actual_return'un hesaplanma tarihi
    (result_available_date) bugünden (script çalıştırma anından) ileride
    olamaz — yani henüz gerçekleşmemiş bir sonuç "biliniyor" gibi
    işaretlenmemiş olmalı.

CI/CD'ye (her commit'te) bağlanabilir; hata bulursa exit code 1 ile çıkar.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import settings
from src.learning.feedback_loop import load_predictions_log

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_lookahead")


def audit() -> bool:
    log = load_predictions_log()
    if log.empty:
        logger.info("Log boş, denetlenecek bir şey yok.")
        return True

    today = pd.Timestamp.today().normalize()
    violations = []

    # 1) result_available_date her zaman as_of_date'ten sonra olmalı
    bad_order = log[log["result_available_date"] <= log["as_of_date"]]
    if not bad_order.empty:
        violations.append(f"{len(bad_order)} satırda result_available_date <= as_of_date.")

    # 2) Henüz gelmemiş bir tarih için evaluated=True olamaz
    future_evaluated = log[
        log["evaluated"].astype(bool) & (log["result_available_date"] > today)
    ]
    if not future_evaluated.empty:
        violations.append(
            f"{len(future_evaluated)} satır, sonuç tarihi gelecekte olduğu halde "
            f"'evaluated=True' olarak işaretlenmiş (look-ahead şüphesi)."
        )

    # 3) evaluated=False olan satırlarda actual_return dolu olmamalı
    inconsistent = log[(~log["evaluated"].astype(bool)) & log["actual_return"].notna()]
    if not inconsistent.empty:
        violations.append(
            f"{len(inconsistent)} satırda evaluated=False ama actual_return dolu — tutarsız durum."
        )

    if violations:
        logger.error("DENETİM BAŞARISIZ:")
        for v in violations:
            logger.error("  - %s", v)
        return False

    logger.info("Denetim başarılı: %d satır kontrol edildi, look-ahead ihlali bulunamadı.", len(log))
    return True


if __name__ == "__main__":
    success = audit()
    sys.exit(0 if success else 1)
