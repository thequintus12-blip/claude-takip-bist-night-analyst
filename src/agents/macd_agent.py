from __future__ import annotations

import pandas as pd

from src.agents.base_agent import AgentSignal, BaseAgent


class MACDAgent(BaseAgent):
    """MACD kesişimi ve histogram yönüne göre görüş bildirir."""

    name = "macd_agent"

    def min_required_rows(self) -> int:
        return self.params.get("macd_slow", 26) + self.params.get("macd_signal", 9) + 5

    def _compute_signal(self, ticker: str, safe_data: pd.DataFrame, as_of_date) -> AgentSignal:
        last = safe_data.iloc[-1]
        prev = safe_data.iloc[-2] if len(safe_data) > 1 else last

        hist = last.get("histogram", float("nan"))
        hist_prev = prev.get("histogram", float("nan"))
        bullish_cross = bool(last.get("macd_bullish_cross", False))
        bearish_cross = bool(last.get("macd_bearish_cross", False))

        if pd.isna(hist):
            return AgentSignal(
                agent=self.name, ticker=ticker, as_of_date=str(pd.Timestamp(as_of_date).date()),
                signal="BEKLE", signal_value=0.0, confidence=0.0, reasoning="MACD hesaplanamadı.",
            )

        histogram_rising = pd.notna(hist_prev) and hist > hist_prev

        if bullish_cross:
            value, conf, signal = 0.75, 0.7, "AL"
            reasoning = "MACD sinyal çizgisini yukarı kesti (bullish cross)."
        elif bearish_cross:
            value, conf, signal = -0.75, 0.7, "SAT"
            reasoning = "MACD sinyal çizgisini aşağı kesti (bearish cross)."
        elif hist > 0 and histogram_rising:
            value, conf, signal = 0.45, 0.5, "BEKLE"
            reasoning = "MACD pozitif ve histogram güçleniyor."
        elif hist > 0 and not histogram_rising:
            value, conf, signal = 0.15, 0.35, "BEKLE"
            reasoning = "MACD pozitif ama momentum zayıflıyor."
        elif hist < 0 and not histogram_rising:
            value, conf, signal = -0.45, 0.5, "BEKLE"
            reasoning = "MACD negatif ve histogram zayıflamaya devam ediyor."
        else:
            value, conf, signal = -0.1, 0.3, "BEKLE"
            reasoning = "MACD negatif ama toparlanma sinyali var."

        return AgentSignal(
            agent=self.name, ticker=ticker, as_of_date=str(pd.Timestamp(as_of_date).date()),
            signal=signal, signal_value=self.clip_signal_value(value), confidence=conf,
            reasoning=reasoning, extra={"histogram": float(hist)},
        )
