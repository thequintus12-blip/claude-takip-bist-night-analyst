"""
Opsiyonel e-posta bildirim modülü.

Bildirim gönderimi GitHub Actions üzerindeki dawidd6/action-send-mail
adımı tarafından yapılır (bkz. .github/workflows/nightly_analysis.yml).
Bu Python modülü, script içinden doğrudan (yerel çalıştırmada) basit bir
özet metni döndürmek için kullanılır; gerçek gönderim Actions tarafına
bırakılmıştır — böylece uygulama kodu SMTP kimlik bilgisi taşımaz.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def notify_summary(summary: str, top_buys: list[str] | None = None, cycle_summary: str | None = None) -> None:
    """Yerel çalıştırmada sadece log'a yazar; e-posta gönderimi GitHub
    Actions adımı tarafından yapılır."""
    message = f"BIST Gece Taraması\n\n{summary}"
    if top_buys:
        message += "\n\nÖne çıkan AL sinyalleri:\n" + "\n".join(f"  - {t}" for t in top_buys)
    if cycle_summary:
        message += "\n\n" + cycle_summary
    logger.info(message)
