from __future__ import annotations

import pandas as pd

from src.agents.base_agent import AgentSignal, BaseAgent


class TrendAgent(BaseAgent):
    """MA50/MA200 yapısı ve ADX trend gücüne göre görüş bildirir."""

    name = "trend_agent"

    def min_required_rows(self) -> int:
        return self.params.get("ma_long", 200) + 5

    def _compute_signal(self, ticker: str, safe_data: pd.DataFrame, as_of_date) -> AgentSignal:
        last = safe_data.iloc[-1]
        adx_threshold = self.params.get("adx_trend_threshold", 20)

        trend_up = bool(last.get("trend_up", False))
        adx_val = last.get("adx", float("nan"))
        strong_trend = pd.notna(adx_val) and adx_val >= adx_threshold

        if trend_up and strong_trend:
            value, conf = 0.8, min(1.0, adx_val / 50)
            reasoning = f"MA50>MA200 ve güçlü trend (ADX={adx_val:.1f})."
            signal = "AL"
        elif trend_up and not strong_trend:
            # Kademeli: ADX eşiğe (adx_threshold) ne kadar yakınsa, "zayıf
            # ama var olan" yükseliş eğilimi o kadar güvenilir sayılır.
            # Eskiden ADX=0.1 ile ADX=19.9 aynı sabit puanı alıyordu.
            proximity = min(1.0, adx_val / adx_threshold) if pd.notna(adx_val) and adx_threshold > 0 else 0.0
            value = 0.15 + 0.2 * proximity
            conf = 0.25 + 0.25 * proximity
            reasoning = f"MA50>MA200 ama trend zayıf (ADX={adx_val:.1f})."
            signal = "BEKLE"
        elif not trend_up and strong_trend:
            value, conf = -0.7, min(1.0, adx_val / 50)
            reasoning = f"MA50<MA200 ve güçlü düşüş trendi (ADX={adx_val:.1f})."
            signal = "SAT"
        else:
            # Aynı kademeli mantık düşüş tarafı için: ADX~0 gerçekten yatay
            # piyasa, eşiğe yakın ADX ise zayıf ama belirgin bir düşüş
            # eğilimi anlamına gelir.
            proximity = min(1.0, adx_val / adx_threshold) if pd.notna(adx_val) and adx_threshold > 0 else 0.0
            value = -0.1 - 0.15 * proximity
            conf = 0.2 + 0.2 * proximity
            reasoning = (
                f"MA50<MA200, zayıf düşüş eğilimi (ADX={adx_val:.1f})."
                if proximity > 0.5 else "Trend belirsiz / yatay piyasa."
            )
            signal = "BEKLE"

        return AgentSignal(
            agent=self.name, ticker=ticker, as_of_date=str(pd.Timestamp(as_of_date).date()),
            signal=signal, signal_value=self.clip_signal_value(value), confidence=conf,
            reasoning=reasoning, extra={"adx": float(adx_val) if pd.notna(adx_val) else None},
        )
