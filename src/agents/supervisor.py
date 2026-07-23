"""
Supervisor / orchestrator.

Not: Orijinal tasarımda LangGraph önerilmişti, ancak deterministik ve
döngüsüz (agent'lar birbirine bağımlı değil, hepsi paralel çalışıp tek bir
birleştirme adımından geçiyor) bir akış için LangGraph gereksiz bir
bağımlılık + Streamlit Cloud'da ek kurulum riski getiriyor. Bu yüzden basit,
saf Python orchestration tercih edildi — önceki projelerdeki "minimum
bağımlılık" felsefesiyle tutarlı. İleride agent'lar arası gerçek
diyalog/iterasyon gerekirse LangGraph bu modülün yerine kolayca geçebilir.
"""

from __future__ import annotations

import pandas as pd

from config import settings
from src.agents.base_agent import AgentSignal, BaseAgent, get_data_as_of
from src.agents.confirmation_agent import ConfirmationAgent
from src.agents.macd_agent import MACDAgent
from src.agents.pattern_agent import PatternAgent
from src.agents.rsi_agent import RSIAgent
from src.agents.trend_agent import TrendAgent
from src.agents.volume_agent import VolumeAgent

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "trend_agent": TrendAgent,
    "rsi_agent": RSIAgent,
    "macd_agent": MACDAgent,
    "volume_agent": VolumeAgent,
    "pattern_agent": PatternAgent,
}


def build_agents(params: dict | None = None) -> dict[str, BaseAgent]:
    return {name: cls(params=params) for name, cls in AGENT_REGISTRY.items()}


def compute_regime_multiplier(benchmark_features: pd.DataFrame | None, as_of_date) -> tuple[float, str]:
    """BIST100 rejim filtresi: piyasa genel trendi düşüşteyse, AL sinyallerini
    bastırır (çarpan < 1.0); yükselişteyse normal/güçlendirilmiş bırakır.

    NOT (kalibrasyon): Eskiden ADX=20'de sert bir eşik vardı — ADX=19.9 ile
    ADX=20.1 tamamen farklı sabit çarpanlara (1.0 vs 1.15, ya da 0.85 vs
    0.6) sıçrıyordu. Artık trend gücü (ADX) ile SÜREKLİ (kademeli) olarak
    ölçekleniyor; ani sıçrama yok."""
    if benchmark_features is None or benchmark_features.empty:
        return 1.0, "Benchmark verisi yok, rejim filtresi uygulanmadı."

    as_of_date = pd.Timestamp(as_of_date)
    safe = benchmark_features.loc[:as_of_date]
    if safe.empty:
        return 1.0, "Benchmark verisi yetersiz."

    last = safe.iloc[-1]
    trend_up = bool(last.get("trend_up", True))
    adx_val = last.get("adx", float("nan"))
    trend_strength = min(1.0, adx_val / 50) if pd.notna(adx_val) else 0.0

    if trend_up:
        # 1.0 (zayıf/yatay yükseliş) -> 1.15 (çok güçlü yükseliş), kademeli
        multiplier = 1.0 + 0.15 * trend_strength
        note = (
            f"XU100 güçlü yükseliş trendinde (ADX={adx_val:.1f}) — sinyaller güçlendirildi."
            if trend_strength > 0.5 else "XU100 yükseliş trendinde."
        )
    else:
        # 0.85 (zayıf/yatay düşüş) -> 0.6 (çok güçlü düşüş), kademeli
        multiplier = 0.85 - 0.25 * trend_strength
        note = (
            f"XU100 güçlü düşüş trendinde (ADX={adx_val:.1f}) — AL sinyalleri bastırıldı."
            if trend_strength > 0.5 else "XU100 yatay/zayıf trendde."
        )
    return round(multiplier, 4), note


def _signal_from_score(score: float) -> str:
    if score >= settings.SIGNAL_BUY_THRESHOLD:
        return "AL"
    if score <= settings.SIGNAL_SELL_THRESHOLD:
        return "SAT"
    return "BEKLE"


def _compute_stop_target(last_row: pd.Series, signal: str, params: dict | None = None) -> dict:
    """ATR bazlı stop/hedef hesaplar. Hedef, trend gücüne (ADX, zaten
    hesaplı) göre ölçeklenir: güçlü trendde hedef daha uzağa konur, zayıf
    trendde daha yakın tutulur. Bu sayede R:R hisseden hisseye, günden
    güne GERÇEKTEN değişir -- eskiden sabit çarpanlar yüzünden R:R her
    zaman tam olarak 2.0 çıkıyordu ve ConfirmationAgent'ın R:R kontrolü
    hiçbir ayırt edici güce sahip değildi."""
    params = params or {}
    close = last_row.get("Close", float("nan"))
    atr_val = last_row.get("atr", float("nan"))
    if pd.isna(close) or pd.isna(atr_val) or signal == "BEKLE":
        return {"stop": None, "target": None, "risk_reward": None}

    stop_mult = params.get("atr_stop_multiplier", 1.5)
    base_target_mult = params.get("atr_target_multiplier_base", 2.0)
    trend_bonus_mult = params.get("atr_target_trend_bonus", 2.0)

    adx_val = last_row.get("adx", float("nan"))
    trend_strength = min(1.0, adx_val / 50) if pd.notna(adx_val) else 0.5  # veri yoksa nötr varsay
    target_mult = base_target_mult + trend_bonus_mult * trend_strength

    if signal == "AL":
        stop = close - stop_mult * atr_val
        target = close + target_mult * atr_val
    else:  # SAT -> kısa pozisyon perspektifiyle referans seviye (otomatik işlem yapılmaz)
        stop = close + stop_mult * atr_val
        target = close - target_mult * atr_val

    risk = abs(close - stop)
    reward = abs(target - close)
    rr = round(reward / risk, 2) if risk > 0 else None

    return {"stop": round(stop, 2), "target": round(target, 2), "risk_reward": rr}


def analyze_ticker(
    ticker: str,
    features: pd.DataFrame,
    as_of_date,
    weights: dict[str, float],
    agents: dict[str, BaseAgent],
    benchmark_features: pd.DataFrame | None = None,
    params: dict | None = None,
) -> dict:
    """Tek bir hisse için tüm agent'ları çalıştırır, ağırlıklı skor üretir.

    Performans notu: `features`, burada BİR KEZ as_of_date'e göre dilimlenir
    (`safe_features`) ve her agent'a bu HAZIR dilim geçirilir. Her agent
    kendi içinde yine get_data_as_of() çağırır (bkz. base_agent.py — bu
    yapısal güvenlik kapısı kaldırılmadı), ama veri zaten sınırlı olduğu
    için bu çağrılar artık gerçek bir dilimleme/kopyalama yapmadan anında
    döner. Önceden her agent, tam (dilimlenmemiş) `features`'ı alıp kendi
    başına dilimliyordu — aynı (hisse, tarih) için 5 kez tekrarlanan bu
    işlem, genetik optimizasyonda ölçülen ana performans darboğazıydı.
    """
    params = params or {}
    as_of_ts = pd.Timestamp(as_of_date)
    safe_features = get_data_as_of(features, as_of_ts)

    agent_outputs: list[AgentSignal] = []
    for name, agent in agents.items():
        result = agent.analyze(ticker, safe_features, as_of_ts)
        result.weight = weights.get(name, 1.0 / len(agents))
        agent_outputs.append(result)

    weighted_sum = sum(a.signal_value * a.confidence * a.weight for a in agent_outputs)
    total_weight = sum(a.weight for a in agent_outputs) or 1.0
    raw_score = weighted_sum / total_weight

    # Çelişki tespiti: agent'lar arasında yön uyuşmazlığı var mı? Eskiden
    # bu bilgi sadece ekranda "⚠️" göstermek için hesaplanıyordu, final
    # skoru hiç etkilemiyordu; sonra sabit bir ceza çarpanı eklendi ama o
    # da çelişkinin ŞİDDETİNE bakmıyordu (+0.15/-0.15 hafif çelişki ile
    # +0.9/-0.9 şiddetli çelişki AYNI cezayı alıyordu). Artık ceza,
    # zıt görüşlerin ne kadar birbirinden uzak olduğuyla orantılı:
    # çelişki ne kadar şiddetliyse skor o kadar küçülür.
    positive_vals = [a.signal_value for a in agent_outputs if a.signal_value > 0.1]
    negative_vals = [a.signal_value for a in agent_outputs if a.signal_value < -0.1]
    has_conflict = bool(positive_vals) and bool(negative_vals)

    if has_conflict:
        severity = min(1.0, (max(positive_vals) - min(negative_vals)) / 2.0)
        max_penalty = 1.0 - params.get("conflict_penalty", 0.8)  # ör. 0.8 -> en fazla %20 ceza
        conflict_mult = 1.0 - max_penalty * severity
    else:
        conflict_mult = 1.0

    regime_mult, regime_note = compute_regime_multiplier(benchmark_features, as_of_date)
    final_score = max(-1.0, min(1.0, raw_score * regime_mult * conflict_mult))

    final_signal = _signal_from_score(final_score)

    last_row = safe_features.iloc[-1] if not safe_features.empty else pd.Series(dtype=float)
    stop_target = _compute_stop_target(last_row, final_signal, params)

    # Likidite (20 günlük ort. TL cinsinden işlem hacmi) — ConfirmationAgent
    # ile AYNI formül, ama artık TÜM hisseler için (sadece AL sinyali
    # verenler için değil) hesaplanıp rapora ekleniyor. Bu, Streamlit'te
    # "en likit N hisseyi göster" görüntüleme filtresinin veri kaynağı.
    if len(safe_features) >= 20:
        liquidity_series = (safe_features["Close"] * safe_features["Volume"]).rolling(20, min_periods=20).mean()
        avg_liquidity = liquidity_series.iloc[-1]
    else:
        avg_liquidity = float("nan")

    return {
        "ticker": ticker,
        "as_of_date": str(as_of_ts.date()),
        "final_signal": final_signal,
        "final_score": round(final_score, 4),
        "has_conflict": has_conflict,
        "conflict_multiplier": conflict_mult,
        "regime_multiplier": regime_mult,
        "regime_note": regime_note,
        "close": round(float(last_row.get("Close", float("nan"))), 2) if not last_row.empty else None,
        "avg_liquidity_try": round(float(avg_liquidity), 0) if pd.notna(avg_liquidity) else None,
        **stop_target,
        "agent_signals": [a.to_dict() for a in agent_outputs],
    }


def analyze_watchlist(
    features_by_ticker: dict[str, pd.DataFrame],
    as_of_date,
    weights: dict[str, float],
    params: dict | None = None,
    benchmark_features: pd.DataFrame | None = None,
) -> list[dict]:
    agents = build_agents(params)
    results = []
    for ticker, features in features_by_ticker.items():
        try:
            result = analyze_ticker(ticker, features, as_of_date, weights, agents, benchmark_features, params)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            results.append({
                "ticker": ticker, "as_of_date": str(pd.Timestamp(as_of_date).date()),
                "final_signal": "BEKLE", "final_score": 0.0, "error": str(exc),
            })
    return results


def apply_confirmation_gate(
    results: list[dict],
    features_by_ticker: dict[str, pd.DataFrame],
    as_of_date,
    params: dict | None = None,
) -> list[dict]:
    """Sadece final_signal == 'AL' olan sonuçlara ikinci göz doğrulamasını
    uygular; her sonuca 'confirmed' (bool|None), 'confirmation_notes' ve
    'confirmation_checks' alanlarını ekler. AL dışındaki sinyaller
    dokunulmadan geçer (confirmed=None -> "doğrulama bu sinyale uygulanmaz").
    """
    agent = ConfirmationAgent(params)
    for r in results:
        if r.get("final_signal") != "AL":
            r["confirmed"] = None
            r["confirmation_notes"] = ""
            r["confirmation_checks"] = {}
            continue

        features = features_by_ticker.get(r["ticker"])
        if features is None:
            r["confirmed"] = None
            r["confirmation_notes"] = "Doğrulama için veri bulunamadı."
            r["confirmation_checks"] = {}
            continue

        result = agent.review(r["ticker"], features, as_of_date, r)
        r["confirmed"] = result.confirmed
        r["confirmation_notes"] = result.reasoning
        r["confirmation_checks"] = result.checks
    return results
