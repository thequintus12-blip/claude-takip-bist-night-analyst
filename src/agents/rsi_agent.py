from __future__ import annotations

import pandas as pd

from src.agents.base_agent import AgentSignal, BaseAgent


class RSIAgent(BaseAgent):
    """RSI seviyesi ve momentum dönüşüne göre görüş bildirir."""

    name = "rsi_agent"

    def min_required_rows(self) -> int:
        return self.params.get("rsi_period", 14) + 5

    def _compute_signal(self, ticker: str, safe_data: pd.DataFrame, as_of_date) -> AgentSignal:
        oversold = self.params.get("rsi_oversold", 30)
        overbought = self.params.get("rsi_overbought", 70)

        last = safe_data.iloc[-1]
        prev = safe_data.iloc[-2] if len(safe_data) > 1 else last
        rsi_val = last.get("rsi", float("nan"))
        rsi_prev = prev.get("rsi", float("nan"))

        if pd.isna(rsi_val):
            return AgentSignal(
                agent=self.name, ticker=ticker, as_of_date=str(pd.Timestamp(as_of_date).date()),
                signal="BEKLE", signal_value=0.0, confidence=0.0, reasoning="RSI hesaplanamadı.",
            )

        recovering_from_oversold = pd.notna(rsi_prev) and rsi_prev < oversold <= rsi_val
        dropping_from_overbought = pd.notna(rsi_prev) and rsi_prev > overbought >= rsi_val

        if recovering_from_oversold:
            value, conf, signal = 0.7, 0.7, "AL"
            reasoning = f"RSI aşırı satımdan ({rsi_prev:.1f}) toparlanıyor ({rsi_val:.1f})."
        elif rsi_val < oversold:
            # Kademeli skorlama: RSI, aşırı satım eşiğinin ne kadar altındaysa
            # (0'a ne kadar yakınsa) o kadar güçlü bir AL eğilimi. Eskiden
            # RSI=29 ile RSI=5 aynı puanı alıyordu -- aşırılık derecesi
            # tamamen göz ardı ediliyordu.
            extremity = min(1.0, (oversold - rsi_val) / oversold) if oversold > 0 else 0.0
            value = 0.4 + 0.3 * extremity
            conf = 0.5 + 0.2 * extremity
            signal = "BEKLE"
            reasoning = f"RSI aşırı satım bölgesinde ({rsi_val:.1f}), dönüş teyidi bekleniyor."
        elif dropping_from_overbought:
            value, conf, signal = -0.6, 0.65, "SAT"
            reasoning = f"RSI aşırı alımdan ({rsi_prev:.1f}) düşüyor ({rsi_val:.1f})."
        elif rsi_val > overbought:
            # Kademeli skorlama: RSI, aşırı alım eşiğinin ne kadar üstündeyse
            # (100'e ne kadar yakınsa) o kadar güçlü bir SAT eğilimi. Eskiden
            # RSI=71 ile RSI=95 (IEYHO örneğindeki 87.5 gibi) AYNI puanı
            # alıyordu -- güçlü trend + aşırı RSI kombinasyonunun taşıdığı
            # gerçek risk, tek bir sabit değere sıkıştırılıyordu.
            extremity = min(1.0, (rsi_val - overbought) / (100 - overbought)) if overbought < 100 else 0.0
            value = -0.3 - 0.5 * extremity
            conf = 0.4 + 0.4 * extremity
            signal = "SAT" if extremity > 0.5 else "BEKLE"
            reasoning = (
                f"RSI çok aşırı alım bölgesinde ({rsi_val:.1f}) — tükenme rallisi riski."
                if extremity > 0.5
                else f"RSI aşırı alım bölgesinde ({rsi_val:.1f}), yeni pozisyon riskli."
            )
        elif 45 <= rsi_val <= 60:
            value, conf, signal = 0.2, 0.4, "BEKLE"
            reasoning = f"RSI nötr bölgede ({rsi_val:.1f}), net momentum yok."
        else:
            value, conf, signal = 0.0, 0.3, "BEKLE"
            reasoning = f"RSI nötr ({rsi_val:.1f})."

        return AgentSignal(
            agent=self.name, ticker=ticker, as_of_date=str(pd.Timestamp(as_of_date).date()),
            signal=signal, signal_value=self.clip_signal_value(value), confidence=conf,
            reasoning=reasoning, extra={"rsi": float(rsi_val)},
        )
