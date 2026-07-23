"""
İkinci göz doğrulama katmanı (confirmation gate).

Ana 5-agent + supervisor sistemi bir hisseye AL kararı verdikten SONRA
devreye girer; yeni bir oy eklemez, var olan kararı bağımsız, farklı
kriterlerle bir kez daha süzer. Bilinçli olarak "minimum indikatör"
felsefesine sadık kalındı: hiçbir yeni teknik indikatör hesaplanmıyor,
sadece zaten var olan verilerin (Close, Volume, RSI, ATR bazlı R:R)
üzerine üç net karar eşiği kuruluyor.

Neden bu üç kriter (ve neden sadece bunlar):
  1. Likidite  — mevcut hacim agent'ı sadece GÖRECELİ hacim artışına
     bakar ("bugün normalden 1.5x fazla"). İşlem hacmi TL bazında zaten
     düşükse (ince piyasa), göreceli artış bile anlamsızdır — stop/hedef
     gerçekçi olmaz, slippage riski yüksektir.
  2. Risk/Ödül — supervisor zaten ATR bazlı stop/hedef hesaplıyor ama
     bunu bir KARAR kriteri olarak kullanmıyordu, sadece bilgi amaçlı
     gösteriyordu. Bu katman, hesaplanmış R:R'yi gerçek bir eşiğe bağlar.
  3. Aşırı RSI vetosu — birincil sistemde RSI sadece 5 oydan biri; çok
     güçlü bir trend (yüksek ADX) + aşırı yüksek RSI kombinasyonu
     ("tükenme rallisi" riski) diğer agent'lar tarafından ezilebiliyor.
     Bu katman buna kesin bir veto gücü tanır.

Look-ahead disiplini: review() içindeki tüm hesaplamalar, diğer agent'larla
aynı get_data_as_of() kapısından geçer — as_of_date'ten sonraki hiçbir
satır bu hesaplamalara karışamaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.agents.base_agent import get_data_as_of


@dataclass
class ConfirmationResult:
    ticker: str
    confirmed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "confirmed": self.confirmed,
            "checks": self.checks,
            "reasoning": self.reasoning,
        }


class ConfirmationAgent:
    """AL sinyali veren hisseleri, birincil sistemden bağımsız üç ek
    kriterle bir kez daha süzer. Oy vermez — sadece onaylar/reddeder."""

    MIN_REQUIRED_ROWS = 25  # 20 günlük likidite penceresi + pay

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def review(self, ticker: str, features: pd.DataFrame, as_of_date,
               supervisor_result: dict) -> ConfirmationResult:
        safe_data = get_data_as_of(features, as_of_date)

        if len(safe_data) < self.MIN_REQUIRED_ROWS:
            return ConfirmationResult(
                ticker=ticker, confirmed=False,
                checks={"yeterli_veri": False},
                reasoning=f"Yetersiz veri ({len(safe_data)}/{self.MIN_REQUIRED_ROWS} gün) — doğrulama yapılamadı.",
            )

        last = safe_data.iloc[-1]

        # 1) Likidite: 20 günlük ortalama TL cinsinden işlem hacmi
        min_liquidity = self.params.get("min_liquidity_try", 5_000_000.0)
        liquidity_series = (safe_data["Close"] * safe_data["Volume"]).rolling(20, min_periods=20).mean()
        liquidity_try = liquidity_series.iloc[-1]
        passed_liquidity = bool(pd.notna(liquidity_try) and liquidity_try >= min_liquidity)

        # 2) Risk/Ödül: supervisor'ın zaten hesapladığı ATR bazlı R:R
        min_rr = self.params.get("min_risk_reward", 1.5)
        rr = supervisor_result.get("risk_reward")
        passed_rr = bool(rr is not None and rr >= min_rr)

        # 3) Aşırı RSI vetosu
        rsi_veto_level = self.params.get("extreme_rsi_veto", 85)
        rsi_val = last.get("rsi", float("nan"))
        passed_rsi = bool(pd.isna(rsi_val) or rsi_val < rsi_veto_level)

        checks = {"likidite": passed_liquidity, "risk_odul": passed_rr, "asiri_rsi": passed_rsi}
        confirmed = all(checks.values())

        reasons = []
        if not passed_liquidity:
            reasons.append(
                f"Likidite düşük (20g ort. {liquidity_try:,.0f} TL < {min_liquidity:,.0f} TL eşiği)."
                if pd.notna(liquidity_try) else "Likidite hesaplanamadı."
            )
        if not passed_rr:
            reasons.append(f"Risk/Ödül yetersiz (R:R={rr} < {min_rr})." if rr is not None else "R:R hesaplanamadı.")
        if not passed_rsi:
            reasons.append(f"RSI aşırı yüksek ({rsi_val:.1f} ≥ {rsi_veto_level}) — tükenme rallisi riski.")

        reasoning = "Tüm kriterler sağlandı, sinyal doğrulandı." if confirmed else " ".join(reasons)

        return ConfirmationResult(ticker=ticker, confirmed=confirmed, checks=checks, reasoning=reasoning)
