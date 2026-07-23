from __future__ import annotations

import pandas as pd

from src.agents.base_agent import AgentSignal, BaseAgent


class VolumeAgent(BaseAgent):
    """Relative volume ve fiyat-hacim teyidine göre görüş bildirir."""

    name = "volume_agent"

    def min_required_rows(self) -> int:
        return self.params.get("rel_volume_period", 20) + 5

    def _compute_signal(self, ticker: str, safe_data: pd.DataFrame, as_of_date) -> AgentSignal:
        threshold = self.params.get("rel_volume_threshold", 1.5)
        last = safe_data.iloc[-1]

        rel_vol = last.get("rel_volume", float("nan"))
        is_green = bool(last.get("is_green", False))

        if pd.isna(rel_vol):
            return AgentSignal(
                agent=self.name, ticker=ticker, as_of_date=str(pd.Timestamp(as_of_date).date()),
                signal="BEKLE", signal_value=0.0, confidence=0.0, reasoning="Hacim verisi yetersiz.",
            )

        high_volume = rel_vol >= threshold

        if high_volume and is_green:
            value, conf, signal = 0.7, min(1.0, rel_vol / (threshold * 2)), "AL"
            reasoning = f"Yüksek hacimle ({rel_vol:.1f}x) yeşil mum — kurumsal ilgi teyidi."
        elif high_volume and not is_green:
            value, conf, signal = -0.6, min(1.0, rel_vol / (threshold * 2)), "SAT"
            reasoning = f"Yüksek hacimle ({rel_vol:.1f}x) kırmızı mum — satış baskısı."
        elif rel_vol < 0.7:
            value, conf, signal = 0.0, 0.3, "BEKLE"
            reasoning = f"Düşük hacim ({rel_vol:.1f}x) — ilgi zayıf, sinyal güvenilir değil."
        else:
            value, conf, signal = 0.1 if is_green else -0.1, 0.3, "BEKLE"
            reasoning = f"Normal hacim seviyesi ({rel_vol:.1f}x)."

        return AgentSignal(
            agent=self.name, ticker=ticker, as_of_date=str(pd.Timestamp(as_of_date).date()),
            signal=signal, signal_value=self.clip_signal_value(value), confidence=conf,
            reasoning=reasoning, extra={"rel_volume": float(rel_vol)},
        )
