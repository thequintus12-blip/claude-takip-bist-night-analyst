from __future__ import annotations

import pandas as pd

from src.agents.base_agent import AgentSignal, BaseAgent


class PatternAgent(BaseAgent):
    """BB squeeze, resistance breakout ve mum kalitesine göre görüş bildirir."""

    name = "pattern_agent"

    def min_required_rows(self) -> int:
        return 65  # bb_squeeze_percentile 60 günlük lookback kullanıyor

    def _compute_signal(self, ticker: str, safe_data: pd.DataFrame, as_of_date) -> AgentSignal:
        last = safe_data.iloc[-1]

        squeeze_pct = last.get("bb_squeeze_pct", float("nan"))
        breakout = bool(last.get("resistance_breakout", False))
        candle_q = last.get("candle_quality", float("nan"))
        is_green = bool(last.get("is_green", False))

        was_squeezed = pd.notna(squeeze_pct) and squeeze_pct <= 0.15
        good_candle = pd.notna(candle_q) and candle_q >= 0.6

        if breakout and was_squeezed and good_candle and is_green:
            value, conf, signal = 0.9, 0.85, "AL"
            reasoning = "Sıkışmadan sonra güçlü mumla direnç kırılımı — yüksek kaliteli setup."
        elif breakout and good_candle and is_green:
            value, conf, signal = 0.6, 0.6, "AL"
            reasoning = "Direnç kırılımı, kaliteli yeşil mumla teyitli."
        elif breakout and not good_candle:
            value, conf, signal = 0.2, 0.35, "BEKLE"
            reasoning = "Direnç kırıldı ama mum kalitesi zayıf (uzun fitil), teyit zayıf."
        elif was_squeezed and not breakout:
            value, conf, signal = 0.15, 0.4, "BEKLE"
            reasoning = "BB sıkışması var ama henüz kırılım yok — izlemede kalsın."
        elif not is_green and pd.notna(candle_q) and candle_q >= 0.6:
            value, conf, signal = -0.4, 0.45, "BEKLE"
            reasoning = "Güçlü kırmızı mum (kaliteli düşüş gövdesi)."
        else:
            value, conf, signal = 0.0, 0.25, "BEKLE"
            reasoning = "Belirgin bir formasyon sinyali yok."

        return AgentSignal(
            agent=self.name, ticker=ticker, as_of_date=str(pd.Timestamp(as_of_date).date()),
            signal=signal, signal_value=self.clip_signal_value(value), confidence=conf,
            reasoning=reasoning,
            extra={
                "bb_squeeze_pct": float(squeeze_pct) if pd.notna(squeeze_pct) else None,
                "breakout": breakout,
            },
        )
