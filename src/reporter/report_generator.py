"""
Gece taraması sonuçlarından okunabilir rapor (DataFrame + özet metinler)
üretir. Streamlit arayüzü bu modülün çıktısını kullanır.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _json_safe(obj):
    """numpy/pandas tiplerini (float64, int64, bool_, Timestamp) standart
    json modülünün serileştirebileceği yerel Python tiplerine çevirir.
    Bu dönüşüm olmadan supervisor/agent çıktılarındaki numpy skalerleri
    (örn. pandas Series'ten .get() ile gelen değerler) json.dump'ı
    'Object of type float64 is not JSON serializable' hatasıyla düşürür."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_full_results(results: list[dict], path: str | Path) -> None:
    """Supervisor'ın tam çıktısını (her hissenin agent_signals detayı
    dahil) JSON olarak kaydeder. Streamlit'teki 'Hisse Bazında Agent
    Detayı' bölümü bu dosyayı okuyarak dolar — bu fonksiyon çağrılmazsa
    o bölüm hep boş kalır."""
    safe_results = _json_safe(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_results, f, ensure_ascii=False, indent=2)


def load_full_results(path: str | Path) -> dict[str, dict]:
    """Kaydedilmiş tam sonuçları, hisse kodu (.IS soneki olmadan) -> sonuç
    sözlüğü şeklinde döndürür. Dosya yoksa (örn. bu güncellemeden önceki
    bir gece taraması) boş sözlük döner."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["ticker"].replace(".IS", ""): r for r in data}


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        confirmed = r.get("confirmed")
        if confirmed is True:
            confirm_label = "✅ Onaylandı"
        elif confirmed is False:
            confirm_label = "❌ Reddedildi"
        else:
            confirm_label = ""

        rows.append({
            "Hisse": r["ticker"].replace(".IS", ""),
            "Sinyal": r.get("final_signal"),
            "Skor": r.get("final_score"),
            "Kapanış": r.get("close"),
            "Likidite (TL)": r.get("avg_liquidity_try"),
            "Stop": r.get("stop"),
            "Hedef": r.get("target"),
            "R:R": r.get("risk_reward"),
            "İkinci Onay": confirm_label,
            "Onay Notu": r.get("confirmation_notes", ""),
            "Çelişki": "⚠️" if r.get("has_conflict") else "",
            "Rejim Notu": r.get("regime_note"),
            "Tarih": r.get("as_of_date"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        signal_order = {"AL": 0, "BEKLE": 1, "SAT": 2}
        df["_sort"] = df["Sinyal"].map(signal_order).fillna(1)
        df = df.sort_values(["_sort", "Skor"], ascending=[True, False]).drop(columns="_sort")
    return df.reset_index(drop=True)


def agent_breakdown_dataframe(result: dict) -> pd.DataFrame:
    """Tek bir hisse için agent bazlı detay tablosu (şeffaflık amaçlı)."""
    rows = []
    for sig in result.get("agent_signals", []):
        rows.append({
            "Agent": sig["agent"],
            "Sinyal": sig["signal"],
            "Değer": sig["signal_value"],
            "Güven": sig["confidence"],
            "Ağırlık": sig["weight"],
            "Açıklama": sig["reasoning"],
        })
    return pd.DataFrame(rows)


def summary_text(results: list[dict]) -> str:
    df = results_to_dataframe(results)
    if df.empty:
        return "Bu gece için sonuç üretilemedi."
    n_buy = (df["Sinyal"] == "AL").sum()
    n_sell = (df["Sinyal"] == "SAT").sum()
    n_wait = (df["Sinyal"] == "BEKLE").sum()
    n_conflict = (df["Çelişki"] == "⚠️").sum()
    date = df["Tarih"].iloc[0] if not df.empty else ""
    return (
        f"{date} taraması: {n_buy} AL, {n_sell} SAT, {n_wait} BEKLE sinyali "
        f"({n_conflict} hissede agent'lar arası görüş ayrılığı var)."
    )
