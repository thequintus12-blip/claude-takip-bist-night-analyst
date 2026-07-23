"""
NOT (bilinçli tasarım kararı): Bu proje "otomatik al-sat ajanları"
oluşturmuyor — sadece raporlama yapan, kendi tahmin başarısına göre
parametrelerini iyileştiren analiz ajanları üretiyor. Bu yüzden RL
(Reinforcement Learning) tabanlı bir trading environment şu an aktif
KULLANILMIYOR; öğrenme tamamen src/learning/feedback_loop.py (skor-bazlı
ağırlıklandırma) ve src/learning/genetic_optimizer.py (walk-forward
parametre araması) üzerinden yürüyor.

Bu dosya, ileride (örn. "agent'ların kendi pozisyon büyüklüğü önerisini de
öğrenmesi" gibi) bir ihtiyaç doğarsa kullanılmak üzere iskelet olarak
bırakılmıştır. Gymnasium kurulumu gerektirir (requirements.txt'de opsiyonel
bölümde belirtilmiştir, varsayılan kurulumda DEĞİLDİR).
"""

from __future__ import annotations


class BistSwingEnvironmentPlaceholder:
    """Aktif değil. Gelecekte gymnasium.Env'den türetilecek bir RL ortamı
    için yer tutucu. Şu an instantiate edilmesi NotImplementedError fırlatır.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RL ortamı bu proje kapsamında aktif değil. Öğrenme mekanizması "
            "için src/learning/feedback_loop.py ve genetic_optimizer.py "
            "kullanılmaktadır."
        )
