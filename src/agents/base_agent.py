"""
Tüm agent'ların türediği temel sınıf.

KRİTİK TASARIM KARARI (Look-Ahead Önleme):
Agent'lara ham feature DataFrame'i ASLA doğrudan verilmez. Bunun yerine
get_data_as_of() ile "as-of-date" sınırlı bir kopya verilir — agent'ın
kodu, fiziksel olarak elinde olmayan satırlara (gelecek günlere) erişemez.
Bu, "agent kodunda yanlışlıkla df.iloc[t+1:] kullanma" gibi insan hatalarına
karşı yapısal bir savunma katmanıdır.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


def get_data_as_of(df: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """DataFrame'i `as_of_date` dahil olmak üzere, ondan SONRAKİ hiçbir
    satırı içermeyecek şekilde keser. Bu fonksiyon tüm agent çağrılarının
    önünde zorunlu bir kapı (gate) olarak kullanılır.

    Performans notu: `df` zaten `as_of_date`'i aşmıyorsa (örn. bir üst
    seviyede -- analyze_ticker() -- bir kez dilimlenip agent'lara aynı
    dilim tekrar geçirildiğinde) hiçbir dilimleme/kopyalama yapmadan
    aynı nesneyi döndürür. Bu kısa devre, aynı (hisse, tarih) için 5-6
    agent'ın her birinin bağımsız olarak aynı veriyi yeniden kesip
    kopyalamasını önler — genetik optimizasyonda ölçülen en büyük
    performans darboğazı tam olarak buydu (profilde toplam sürenin
    %54'ü burada harcanıyordu). `.copy()` kaldırıldı çünkü hiçbir agent
    aldığı veriyi yerinde değiştirmiyor (salt-okunur kullanım); look-ahead
    güvenliği zaten `.loc[:as_of_date]` diliminin kendisinden geliyor,
    kopyalamadan değil.
    """
    as_of_date = pd.Timestamp(as_of_date)
    max_date = df.index.max()
    if max_date <= as_of_date:
        # Veri zaten as_of_date'i aşmıyor -> dilimlemeye gerek yok.
        return df
    return df.loc[:as_of_date]


@dataclass
class AgentSignal:
    """Tüm agent'ların ürettiği standart çıktı formatı."""
    agent: str
    ticker: str
    as_of_date: str
    signal: str            # "AL" | "SAT" | "BEKLE"
    signal_value: float    # -1.0 (güçlü sat) .. +1.0 (güçlü al)
    confidence: float       # 0.0 - 1.0
    reasoning: str
    weight: float = 1.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "signal": self.signal,
            "signal_value": self.signal_value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "weight": self.weight,
            **self.extra,
        }


class BaseAgent(ABC):
    """Tüm uzman agent'ların temel sınıfı."""

    name: str = "base_agent"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def analyze(self, ticker: str, features: pd.DataFrame, as_of_date) -> AgentSignal:
        """Dışarıya açık tek giriş noktası. Veri burada as-of-date'e göre
        kesilir, agent'ın _compute_signal() metodu ASLA tam DataFrame'i
        görmez."""
        safe_data = get_data_as_of(features, as_of_date)

        min_required = self.min_required_rows()
        if len(safe_data) < min_required:
            return AgentSignal(
                agent=self.name, ticker=ticker, as_of_date=str(pd.Timestamp(as_of_date).date()),
                signal="BEKLE", signal_value=0.0, confidence=0.0,
                reasoning=f"Yetersiz veri ({len(safe_data)}/{min_required} gün).",
            )

        return self._compute_signal(ticker, safe_data, as_of_date)

    @abstractmethod
    def _compute_signal(self, ticker: str, safe_data: pd.DataFrame, as_of_date) -> AgentSignal:
        """Alt sınıflar bunu implemente eder. `safe_data` zaten as-of-date'e
        göre kesilmiştir, içinde gelecek veri YOKTUR."""
        raise NotImplementedError

    def min_required_rows(self) -> int:
        return 50

    def clip_signal_value(self, value: float) -> float:
        return max(-1.0, min(1.0, value))
