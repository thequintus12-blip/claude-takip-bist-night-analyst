"""
Haftalık portföy döngüsü.

Kullanıcının istediği akış:
  - Hafta içinde AL sinyali verip ikinci onaydan (confirmed=True) geçen
    hisseler "gerçekten alınabilir" kabul edilir; bu havuzdan en iyi
    potansiyele sahip (final_score'u en yüksek) olanlar takibe alınır.
  - Haftanın ilk işlem günü (normalde Pazartesi; bir önceki Cuma BIST
    kapalıysa haftanın ilk çalıştığı gün): bir önceki Cuma akşamı
    belirlenmiş "gelecek hafta adayları" varsa onlar takibe alınır; yoksa
    (sistem ilk kez çalışıyorsa) o günün taze analizinden seçilir.
  - Aynı akşam (ve Salı/Çarşamba/Perşembe akşamları) takip edilen
    hisselerin durumu o günün taze verisiyle yeniden değerlendirilir:
    - Bir hisse artık AL sinyali vermiyorsa veya ikinci onaydan
      geçemiyorsa "kötüleşti" sayılır ve yerine (varsa) takip
      edilmeyen en güçlü AL+onaylı aday önerilir.
    - Kötüleşen yoksa bile, takip edilenlerin en zayıfından belirgin
      şekilde (bkz. PORTFOLIO_UPGRADE_MARGIN) daha iyi potansiyele sahip
      takip-dışı bir aday varsa, isteğe bağlı bir "yükseltme" önerisi
      sunulur.
  - Perşembe akşamı, yukarıdaki gözden geçirmeye ek olarak "yarın Cuma
    kapanışta şunlar satılacak" uyarısı eklenir.
  - Cuma akşamı: "bugün kapanışta şunlar satılmalı" hatırlatması yapılır,
    takip listesi boşaltılır (pozisyonlar kapandığı varsayılır) ve o
    günün taze analizinden gelecek haftanın aday listesi belirlenip
    duruma kaydedilir (yani "hafta sonu analizi" fiilen Cuma'nın kendi
    kapanış-sonrası taramasıdır — BIST hafta sonu işlem görmediği için
    Cumartesi/Pazar günü yeni veri gelmez).

Bilinen sınırlama: Sistem BIST resmi tatil takvimini bilmiyor; "Perşembe"
ve "Cuma" tespiti, o günün analiz tarihinin (as_of_date) haftanın kaçıncı
günü olduğuna bakılarak yapılır. Cuma resmi tatilse bu bildirimler bir gün
kayabilir. "Yeni hafta başlangıcı" tespiti ise haftanın hangi gününde
olursa olsun (Pazartesi tatilse Salı günü de olsa) doğru çalışır, çünkü
"bu haftanın Pazartesi tarihi daha önce işlenmiş mi" kontrolüne dayanır.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from config import settings
from src.portfolio import performance_tracker

logger = logging.getLogger(__name__)

DAY_TYPE_LABELS = {
    "hafta_baslangici": "Yeni Hafta Başlangıcı",
    "hafta_ortasi_soguk_baslangic": "Hafta Ortası İlk Başlatma",
    "ara_hafta_gozden_gecirme": "Ara Hafta Gözden Geçirme",
    "persembe_uyarisi": "Perşembe — Cuma Satış Uyarısı",
    "cuma_satisi": "Cuma — Kapanışta Satış",
}

_EMPTY_STATE = {
    "held": [],
    "cycle_week_start": None,
    "next_week_candidates": [],
    "last_run_date": None,
}


def load_state() -> dict:
    if not settings.PORTFOLIO_STATE_FILE.exists():
        return dict(_EMPTY_STATE)
    try:
        with open(settings.PORTFOLIO_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("portfolio_state.json okunamadı, sıfırdan başlanıyor: %s", exc)
        return dict(_EMPTY_STATE)
    merged = dict(_EMPTY_STATE)
    merged.update(state)
    return merged


def save_state(state: dict) -> None:
    settings.PORTFOLIO_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.PORTFOLIO_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _clean_ticker(ticker: str) -> str:
    return ticker.replace(".IS", "")


def _candidate_pool(results: list[dict], exclude: set[str] | None = None) -> list[dict]:
    """AL sinyali verip ikinci onaydan (confirmed=True) geçen hisseleri,
    final_score'a göre (en iyi potansiyel en önde) sıralı döndürür.
    Sadece final_signal=='AL' AND confirmed==True olanlar dahil edilir —
    sistemin kendi kalite eşiğini (ikinci onay) gevşetmeden."""
    exclude = exclude or set()
    pool = []
    for r in results:
        if r.get("final_signal") != "AL" or r.get("confirmed") is not True:
            continue
        ticker = _clean_ticker(r["ticker"])
        if ticker in exclude:
            continue
        pool.append({
            "ticker": ticker,
            "score": float(r.get("final_score", 0.0)),
            "close": r.get("close"),
            "stop": r.get("stop"),
            "target": r.get("target"),
            "risk_reward": r.get("risk_reward"),
        })
    pool.sort(key=lambda x: x["score"], reverse=True)
    return pool


def _make_held_entry(candidate: dict, entered_on: str) -> dict:
    return {
        "ticker": candidate["ticker"],
        "entered_on": entered_on,
        "entry_score": candidate["score"],
        "entry_price": candidate.get("close"),
        "last_score": candidate["score"],
        "last_price": candidate.get("close"),
        "last_signal": "AL",
        "last_confirmed": True,
    }


def build_weekly_cycle_report(
    results: list[dict],
    as_of_date,
    top_n: int | None = None,
    upgrade_margin: float | None = None,
) -> dict:
    """Her gece run_daily_analysis.py tarafından, apply_confirmation_gate
    sonrası çağrılır. Duruma (data/processed/portfolio_state.json) göre
    haftanın hangi aşamasında olunduğunu belirler, gerekli aksiyonu/öneriyi
    üretir ve durumu bir sonraki çalıştırma için kaydeder."""
    top_n = settings.PORTFOLIO_MAX_HOLDINGS if top_n is None else top_n
    upgrade_margin = settings.PORTFOLIO_UPGRADE_MARGIN if upgrade_margin is None else upgrade_margin

    as_of_ts = pd.Timestamp(as_of_date).normalize()
    weekday = as_of_ts.weekday()  # 0=Pazartesi ... 4=Cuma
    monday_of_week = as_of_ts - pd.Timedelta(days=weekday)
    monday_str = str(monday_of_week.date())

    state = load_state()
    by_ticker = {_clean_ticker(r["ticker"]): r for r in results}

    held = list(state.get("held", []))
    messages: list[str] = []
    new_entries: list[str] = []
    day_type = None

    is_new_week = state.get("cycle_week_start") != monday_str
    if is_new_week:
        day_type = "hafta_baslangici"
        pending = state.get("next_week_candidates", [])
        if held:
            messages.append(
                "Not: yeni hafta başlarken önceki listeden boşaltılmamış "
                f"görünen hisseler bulundu ({', '.join(h['ticker'] for h in held)}); "
                "bu hafta başlangıcında yeni liste bunların yerini aldı."
            )
        if pending:
            chosen = pending[:top_n]
            messages.append(
                "Geçen hafta sonu analizinde belirlenen adaylar bu hafta için takibe alındı."
            )
        else:
            chosen = _candidate_pool(results)[:top_n]
            messages.append(
                "Önceki haftadan devreden bir aday listesi bulunmadığı için "
                "takip listesi bugünün analizine göre oluşturuldu."
            )
        held = [_make_held_entry(c, str(as_of_ts.date())) for c in chosen]
        new_entries = [c["ticker"] for c in chosen]
        state["cycle_week_start"] = monday_str
        state["next_week_candidates"] = []

    if not held:
        # Güvenlik ağı: yukarıdaki dal hiç tetiklenmediyse (örn. hafta
        # ortasında ilk kez çalıştırılıyor) ya da yeni hafta başlangıcında
        # hiç aday bulunamadıysa, elde hiçbir şey yokken sistem sessiz
        # kalmasın diye bugünün analizinden taze bir liste dener.
        if day_type is None:
            day_type = "hafta_ortasi_soguk_baslangic"
            messages.append(
                "Takip edilen hisse bulunamadığı için liste bugünün analizine göre oluşturuldu."
            )
            chosen = _candidate_pool(results)[:top_n]
            held = [_make_held_entry(c, str(as_of_ts.date())) for c in chosen]
            new_entries = [c["ticker"] for c in chosen]
            state["cycle_week_start"] = monday_str
        elif not held and not new_entries:
            messages.append(
                "Bugün ikinci onaydan geçen hiçbir AL sinyali olmadığı için "
                "bu hafta takip edilecek yeni bir hisse bulunamadı."
            )

    # ── Akşam gözden geçirme: her hafta içi günü (adoption günü dahil) ────
    held_tickers_now = {h["ticker"] for h in held}
    worsened = []
    for h in held:
        current = by_ticker.get(h["ticker"])
        if current is None:
            worsened.append({"ticker": h["ticker"], "reason": "Bugünkü taramada bu hisse için veri/sonuç bulunamadı."})
            # last_price BİLEREK korunur (dokunulmaz): Cuma'da gerçekleşen
            # getiriyi hesaplayabilmek için en son bilinen fiyata ihtiyaç
            # var — veri bir gün eksikse bunu None'a çevirmek o bilgiyi
            # kalıcı olarak kaybettirir.
            h["last_score"], h["last_signal"], h["last_confirmed"] = None, None, None
            continue
        h["last_score"] = float(current.get("final_score", 0.0))
        h["last_signal"] = current.get("final_signal")
        h["last_confirmed"] = current.get("confirmed")
        if current.get("close") is not None:
            h["last_price"] = current.get("close")
        if current.get("final_signal") != "AL":
            worsened.append({
                "ticker": h["ticker"],
                "reason": f"Sinyal artık AL değil (şu an: {current.get('final_signal')}).",
            })
        elif current.get("confirmed") is not True:
            worsened.append({
                "ticker": h["ticker"],
                "reason": "AL sinyali veriyor ama ikinci onay katmanından artık geçemiyor.",
            })

    worsened_tickers = [w["ticker"] for w in worsened]
    swap_suggestions = []
    alt_pool = _candidate_pool(results, exclude=held_tickers_now)

    # Kötüleşen her hisse için (varsa) ayrı bir alternatif öner.
    for wt in worsened_tickers:
        if not alt_pool:
            break
        best_alt = alt_pool.pop(0)
        target = next(h for h in held if h["ticker"] == wt)
        swap_suggestions.append({
            "sell_ticker": wt,
            "buy_ticker": best_alt["ticker"],
            "sell_score": target["last_score"],
            "buy_score": best_alt["score"],
            "reason": "worsened",
        })

    # Kötüleşen yoksa/kalan varsa: hâlâ iyi durumdaki en zayıf hisseyi,
    # belirgin şekilde daha güçlü kalan bir alternatifle "yükseltme" öner.
    ok_held = [h for h in held if h["ticker"] not in worsened_tickers]
    if alt_pool and ok_held:
        weakest_ok = min(ok_held, key=lambda h: h["last_score"] if h["last_score"] is not None else float("-inf"))
        weakest_score = weakest_ok["last_score"] if weakest_ok["last_score"] is not None else float("-inf")
        best_alt = alt_pool[0]
        if best_alt["score"] - weakest_score >= upgrade_margin:
            swap_suggestions.append({
                "sell_ticker": weakest_ok["ticker"],
                "buy_ticker": best_alt["ticker"],
                "sell_score": weakest_score,
                "buy_score": best_alt["score"],
                "reason": "upgrade",
            })

    liquidation_list: list[str] = []
    next_week_candidates: list[dict] = []
    is_thursday_notice = weekday == 3
    is_friday_liquidation = weekday == 4

    if is_thursday_notice:
        day_type = day_type or "persembe_uyarisi"
        liquidation_list = sorted(held_tickers_now)

    completed_positions: list[dict] = []
    if is_friday_liquidation:
        day_type = "cuma_satisi"
        liquidation_list = sorted(held_tickers_now)
        for h in held:
            entry_price = h.get("entry_price")
            exit_price = h.get("last_price")
            realized_return = None
            if entry_price and exit_price and entry_price > 0:
                realized_return = round(exit_price / entry_price - 1, 4)
            row = {
                "entered_on": h["entered_on"],
                "exited_on": str(as_of_ts.date()),
                "ticker": h["ticker"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "realized_return": realized_return,
                "holding_days": (as_of_ts.date() - pd.Timestamp(h["entered_on"]).date()).days,
                "exit_reason": "friday_liquidation",
            }
            completed_positions.append(row)
        performance_tracker.log_completed_positions(completed_positions)

        next_week_candidates = _candidate_pool(results)[:top_n]
        state["next_week_candidates"] = next_week_candidates
        held = []
        held_tickers_now = set()

    if day_type is None:
        day_type = "ara_hafta_gozden_gecirme"

    state["held"] = held
    state["last_run_date"] = str(as_of_ts.date())
    save_state(state)

    return {
        "as_of_date": str(as_of_ts.date()),
        "day_type": day_type,
        "day_type_label": DAY_TYPE_LABELS.get(day_type, day_type),
        "is_thursday_notice": is_thursday_notice,
        "is_friday_liquidation": is_friday_liquidation,
        "new_entries": new_entries,
        "held_after": sorted(held_tickers_now),
        "worsened": worsened,
        "swap_suggestions": swap_suggestions,
        "liquidation_list": liquidation_list,
        "next_week_candidates": [c["ticker"] for c in next_week_candidates],
        "completed_positions": completed_positions,
        "messages": messages,
    }


def cycle_summary_text(report: dict) -> str:
    """Rapor sözlüğünü, e-posta/log/Streamlit'te gösterilecek okunabilir
    bir Türkçe metne çevirir."""
    lines = [f"📅 Haftalık Döngü — {report['as_of_date']} ({report['day_type_label']})"]

    if report["new_entries"]:
        lines.append(f"🆕 Yeni takibe alınanlar: {', '.join(report['new_entries'])}")

    if report["held_after"]:
        lines.append(f"📌 Şu an takip edilen hisseler: {', '.join(report['held_after'])}")
    else:
        lines.append("📌 Şu an takip edilen hisse yok.")

    for w in report["worsened"]:
        lines.append(f"⚠️ {w['ticker']}: {w['reason']}")

    for s in report["swap_suggestions"]:
        if s["reason"] == "worsened":
            verb = "durumu kötüleştiği için"
        else:
            verb = "belirgin şekilde daha güçlü bir alternatif bulunduğu için"
        sell_score = f"{s['sell_score']:.3f}" if s["sell_score"] is not None else "?"
        lines.append(
            f"🔁 Öneri: {s['sell_ticker']} yerine {s['buy_ticker']} değerlendirilebilir "
            f"({verb}; skor {sell_score} → {s['buy_score']:.3f})."
        )

    if report["is_thursday_notice"] and not report["is_friday_liquidation"]:
        if report["liquidation_list"]:
            lines.append(
                f"⏰ Yarın (Cuma) borsa kapanışında şu hisseler satılacak: {', '.join(report['liquidation_list'])}."
            )
        else:
            lines.append("Bu hafta zaten takip edilen bir hisse olmadığı için Cuma'da satılacak bir şey yok.")

    if report["is_friday_liquidation"]:
        if report["liquidation_list"]:
            lines.append(
                f"🔴 Bugün borsa kapanışında şu hisseler satılmalı: {', '.join(report['liquidation_list'])}."
            )
        else:
            lines.append("Bu hafta zaten takip edilen bir hisse yoktu, satılacak bir şey yok.")

        priced = [p for p in report["completed_positions"] if p["realized_return"] is not None]
        if priced:
            detail = ", ".join(f"{p['ticker']} {p['realized_return']:+.2%}" for p in priced)
            avg_ret = sum(p["realized_return"] for p in priced) / len(priced)
            lines.append(f"📊 Bu haftanın gerçekleşen getirisi: {detail} (ortalama: {avg_ret:+.2%}).")

        if report["next_week_candidates"]:
            lines.append(
                "📋 Gelecek hafta (Pazartesi) için aday hisseler: "
                + ", ".join(report["next_week_candidates"])
            )
        else:
            lines.append(
                "Bugün ikinci onaydan geçen bir AL sinyali olmadığı için "
                "gelecek hafta için henüz bir aday belirlenemedi."
            )

    for m in report["messages"]:
        lines.append(f"ℹ️ {m}")

    return "\n".join(lines)
