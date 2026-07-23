"""src/portfolio/cycle_manager.py için testler.

Haftalık döngünün beş temel davranışını doğrular:
  1. Pazartesi (yeni hafta): bekleyen adaylar varsa onlar takibe alınır.
  2. Aynı akşam (veya Salı/Çarşamba) gözden geçirme: kötüleşen bir hisse
     tespit edilip yerine daha iyi bir alternatif önerilir.
  3. Perşembe: ertesi gün (Cuma) satış uyarısı verilir, takip listesi
     BOZULMADAN kalır (henüz satılmadı).
  4. Cuma: satış hatırlatması yapılır, takip listesi boşaltılır ve
     gelecek haftanın adayları belirlenip duruma kaydedilir.
  5. Hafta ortasında (state hiç yoksa) soğuk başlangıç ve aynı gün için
     tekrar çalıştırmanın (idempotency) sonucu değiştirmediği.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import settings
from src.portfolio import cycle_manager, performance_tracker


@pytest.fixture(autouse=True)
def _isolate_state_file(tmp_path, monkeypatch):
    """Her testin kendi geçici state/performans dosyalarını kullanmasını
    sağlar, gerçek repo'daki data/processed/*.json,*.csv'ye dokunulmaz."""
    monkeypatch.setattr(settings, "PORTFOLIO_STATE_FILE", tmp_path / "portfolio_state.json")
    monkeypatch.setattr(settings, "CYCLE_PERFORMANCE_LOG_FILE", tmp_path / "cycle_performance_log.csv")
    yield


def _result(ticker: str, signal: str = "AL", confirmed: bool | None = True, score: float = 0.15) -> dict:
    return {
        "ticker": f"{ticker}.IS",
        "final_signal": signal,
        "final_score": score,
        "confirmed": confirmed,
        "close": 10.0,
        "stop": 9.0,
        "target": 12.0,
        "risk_reward": 2.0,
    }


# Pazartesi=2026-07-20, Salı=21, Çarşamba=22, Perşembe=23, Cuma=24 (bkz. üstteki tarih doğrulaması)
MON, TUE, WED, THU, FRI = (
    pd.Timestamp("2026-07-20"), pd.Timestamp("2026-07-21"), pd.Timestamp("2026-07-22"),
    pd.Timestamp("2026-07-23"), pd.Timestamp("2026-07-24"),
)


def test_weekday_anchor_dates_are_correct():
    assert MON.weekday() == 0 and FRI.weekday() == 4


def test_first_ever_run_with_no_state_is_labeled_week_start():
    # Hiç state dosyası yokken ilk çalıştırma: hangi gün olursa olsun,
    # "bu haftaya ait kayıtlı bir başlangıç yok" durumudur, bu yüzden
    # hafta_baslangici olarak etiketlenir (haftanın hangi gününde olursa
    # olsun takip burada başlar).
    results = [_result("AAA", score=0.20), _result("BBB", score=0.15), _result("CCC", "BEKLE", None, 0.05)]
    report = cycle_manager.build_weekly_cycle_report(results, WED, top_n=5)
    assert report["day_type"] == "hafta_baslangici"
    assert report["held_after"] == ["AAA", "BBB"]
    assert report["new_entries"] == ["AAA", "BBB"]


def test_midweek_empty_held_within_same_week_triggers_cold_start_fallback():
    # cycle_week_start bu haftayla ZATEN eşleşiyor (yani Pazartesi/Salı
    # çalıştı ama elde hiç hisse kalmadı) — bu durumda is_new_week=False
    # olduğu için "hafta_ortasi_soguk_baslangic" güvenlik ağı devreye girer.
    state = {
        "held": [], "cycle_week_start": "2026-07-20",
        "next_week_candidates": [], "last_run_date": "2026-07-21",
    }
    cycle_manager.save_state(state)

    results = [_result("AAA", score=0.20)]
    report = cycle_manager.build_weekly_cycle_report(results, WED, top_n=5)
    assert report["day_type"] == "hafta_ortasi_soguk_baslangic"
    assert report["held_after"] == ["AAA"]


def test_monday_adopts_pending_candidates_from_previous_friday():
    state = {
        "held": [], "cycle_week_start": "2026-07-13",
        "next_week_candidates": [{"ticker": "AAA", "score": 0.22, "close": 10, "stop": 9, "target": 12, "risk_reward": 2.0}],
        "last_run_date": "2026-07-17",
    }
    cycle_manager.save_state(state)

    results = [_result("AAA", score=0.19)]  # Pazartesi taze verisiyle hâlâ AL + onaylı
    report = cycle_manager.build_weekly_cycle_report(results, MON, top_n=5)

    assert report["day_type"] == "hafta_baslangici"
    assert report["new_entries"] == ["AAA"]
    assert report["held_after"] == ["AAA"]
    assert report["worsened"] == []  # Pazartesi'nin taze verisiyle hâlâ iyi durumda


def test_monday_evening_review_flags_worsened_pending_candidate_and_suggests_swap():
    state = {
        "held": [], "cycle_week_start": "2026-07-13",
        "next_week_candidates": [{"ticker": "BBB", "score": 0.20, "close": 10, "stop": 9, "target": 12, "risk_reward": 2.0}],
        "last_run_date": "2026-07-17",
    }
    cycle_manager.save_state(state)

    # BBB Cuma'dan Pazartesi'ye kötüleşmiş (artık AL değil); CCC ise güçlü bir alternatif.
    results = [_result("BBB", signal="BEKLE", confirmed=None, score=0.02), _result("CCC", score=0.21)]
    report = cycle_manager.build_weekly_cycle_report(results, MON, top_n=5)

    assert report["new_entries"] == ["BBB"]  # önce adopte edildi
    assert [w["ticker"] for w in report["worsened"]] == ["BBB"]
    assert report["swap_suggestions"][0]["sell_ticker"] == "BBB"
    assert report["swap_suggestions"][0]["buy_ticker"] == "CCC"
    assert report["swap_suggestions"][0]["reason"] == "worsened"


def test_midweek_worsening_detected_against_currently_held():
    state = {
        "held": [{"ticker": "XXX", "entered_on": "2026-07-20", "entry_score": 0.18,
                   "last_score": 0.18, "last_signal": "AL", "last_confirmed": True}],
        "cycle_week_start": "2026-07-20", "next_week_candidates": [], "last_run_date": "2026-07-21",
    }
    cycle_manager.save_state(state)

    results = [_result("XXX", signal="AL", confirmed=False, score=0.10), _result("YYY", score=0.25)]
    report = cycle_manager.build_weekly_cycle_report(results, TUE, top_n=5)

    assert report["day_type"] == "ara_hafta_gozden_gecirme"
    assert [w["ticker"] for w in report["worsened"]] == ["XXX"]
    assert report["swap_suggestions"][0] == {
        "sell_ticker": "XXX", "buy_ticker": "YYY", "sell_score": 0.10, "buy_score": 0.25, "reason": "worsened",
    }


def test_upgrade_suggestion_when_nothing_worsened_but_better_alternative_exists():
    state = {
        "held": [{"ticker": "XXX", "entered_on": "2026-07-20", "entry_score": 0.13,
                   "last_score": 0.13, "last_signal": "AL", "last_confirmed": True}],
        "cycle_week_start": "2026-07-20", "next_week_candidates": [], "last_run_date": "2026-07-21",
    }
    cycle_manager.save_state(state)

    # XXX hâlâ AL+onaylı (kötüleşmedi) ama YYY belirgin şekilde daha güçlü.
    results = [_result("XXX", score=0.13), _result("YYY", score=0.30)]
    report = cycle_manager.build_weekly_cycle_report(results, WED, top_n=5, upgrade_margin=0.03)

    assert report["worsened"] == []
    assert report["swap_suggestions"][0]["reason"] == "upgrade"
    assert report["swap_suggestions"][0]["buy_ticker"] == "YYY"


def test_no_upgrade_suggestion_when_difference_below_margin():
    state = {
        "held": [{"ticker": "XXX", "entered_on": "2026-07-20", "entry_score": 0.13,
                   "last_score": 0.13, "last_signal": "AL", "last_confirmed": True}],
        "cycle_week_start": "2026-07-20", "next_week_candidates": [], "last_run_date": "2026-07-21",
    }
    cycle_manager.save_state(state)

    results = [_result("XXX", score=0.13), _result("YYY", score=0.14)]  # fark 0.03'ün altında
    report = cycle_manager.build_weekly_cycle_report(results, WED, top_n=5, upgrade_margin=0.03)

    assert report["swap_suggestions"] == []


def test_thursday_gives_advance_notice_without_clearing_holdings():
    state = {
        "held": [{"ticker": "XXX", "entered_on": "2026-07-20", "entry_score": 0.18,
                   "last_score": 0.18, "last_signal": "AL", "last_confirmed": True}],
        "cycle_week_start": "2026-07-20", "next_week_candidates": [], "last_run_date": "2026-07-22",
    }
    cycle_manager.save_state(state)

    results = [_result("XXX", score=0.16)]
    report = cycle_manager.build_weekly_cycle_report(results, THU, top_n=5)

    assert report["is_thursday_notice"] is True
    assert report["liquidation_list"] == ["XXX"]
    assert report["held_after"] == ["XXX"]  # henüz satılmadı, sadece uyarı verildi

    persisted = cycle_manager.load_state()
    assert [h["ticker"] for h in persisted["held"]] == ["XXX"]


def test_friday_liquidates_and_saves_next_week_candidates():
    state = {
        "held": [{"ticker": "XXX", "entered_on": "2026-07-20", "entry_score": 0.18,
                   "last_score": 0.18, "last_signal": "AL", "last_confirmed": True}],
        "cycle_week_start": "2026-07-20", "next_week_candidates": [], "last_run_date": "2026-07-23",
    }
    cycle_manager.save_state(state)

    results = [_result("XXX", signal="BEKLE", confirmed=None, score=0.05), _result("ZZZ", score=0.28)]
    report = cycle_manager.build_weekly_cycle_report(results, FRI, top_n=5)

    assert report["is_friday_liquidation"] is True
    assert report["liquidation_list"] == ["XXX"]  # bu hafta takip edilen, ne olursa olsun satılır
    assert report["held_after"] == []  # pozisyonlar kapandı
    assert report["next_week_candidates"] == ["ZZZ"]  # XXX artık AL/onaylı olmadığı için aday değil

    persisted = cycle_manager.load_state()
    assert persisted["held"] == []
    assert [c["ticker"] for c in persisted["next_week_candidates"]] == ["ZZZ"]


def test_friday_liquidation_allows_still_qualifying_stock_to_reappear_next_week():
    # Cuma'da satılan bir hisse, o gün hâlâ AL+onaylı ise gelecek hafta
    # adayı olarak TEKRAR seçilebilir — haftalık satış bir "ceza" değil,
    # programlı bir kâr/risk yönetimi adımıdır; hâlâ en iyi seçenekse
    # aynı hisse yeniden takibe alınabilir.
    state = {
        "held": [{"ticker": "XXX", "entered_on": "2026-07-20", "entry_score": 0.18,
                   "last_score": 0.18, "last_signal": "AL", "last_confirmed": True}],
        "cycle_week_start": "2026-07-20", "next_week_candidates": [], "last_run_date": "2026-07-23",
    }
    cycle_manager.save_state(state)

    results = [_result("XXX", score=0.28)]
    report = cycle_manager.build_weekly_cycle_report(results, FRI, top_n=5)

    assert report["liquidation_list"] == ["XXX"]
    assert report["next_week_candidates"] == ["XXX"]


def test_rerunning_same_date_is_idempotent():
    results = [_result("AAA", score=0.20)]
    report1 = cycle_manager.build_weekly_cycle_report(results, MON, top_n=5)
    report2 = cycle_manager.build_weekly_cycle_report(results, MON, top_n=5)

    assert report1["held_after"] == report2["held_after"] == ["AAA"]
    # ikinci çalıştırmada artık "yeni hafta" tetiklenmemeli (aynı hafta, tekrar çalıştırma)
    assert report2["day_type"] == "ara_hafta_gozden_gecirme"
    assert report2["new_entries"] == []


def test_no_candidates_available_does_not_crash():
    results = [_result("AAA", signal="BEKLE", confirmed=None, score=0.01)]
    report = cycle_manager.build_weekly_cycle_report(results, WED, top_n=5)

    assert report["held_after"] == []
    assert report["swap_suggestions"] == []
    assert any("bulunamadı" in m for m in report["messages"])


def test_full_week_logs_realized_return_on_friday():
    """Pazartesi'den Cuma'ya kadar bir hisse fiyatı 100 -> 110 giderse,
    Cuma'da gerçekleşen getiri +%10 olarak loglanmalı."""
    prices = {MON: 100.0, TUE: 102.0, WED: 105.0, THU: 108.0, FRI: 110.0}

    for day, price in prices.items():
        results = [_result("AAA", score=0.20)]
        results[0]["close"] = price
        cycle_manager.build_weekly_cycle_report(results, day, top_n=5)

    log = performance_tracker.load_performance_log()
    assert len(log) == 1
    row = log.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["entry_price"] == 100.0
    assert row["exit_price"] == 110.0
    assert row["realized_return"] == pytest.approx(0.10)
    assert row["exit_reason"] == "friday_liquidation"


def test_realized_return_uses_last_known_price_when_friday_data_missing():
    """Bir hisse Perşembe günü veri kaybederse (ör. geçici veri sorunu),
    Cuma'daki gerçekleşen getiri hesaplanırken son BİLİNEN fiyat (Çarşamba)
    kullanılmalı — None'a düşüp veriyi tamamen kaybetmemeli."""
    r_mon = [_result("AAA", score=0.20)]
    r_mon[0]["close"] = 50.0
    cycle_manager.build_weekly_cycle_report(r_mon, MON, top_n=5)

    r_wed = [_result("AAA", score=0.18)]
    r_wed[0]["close"] = 55.0
    cycle_manager.build_weekly_cycle_report(r_wed, WED, top_n=5)

    cycle_manager.build_weekly_cycle_report([], THU, top_n=5)  # AAA bugün veri setinde yok
    cycle_manager.build_weekly_cycle_report([], FRI, top_n=5)  # hâlâ yok

    log = performance_tracker.load_performance_log()
    row = log.iloc[0]
    assert row["entry_price"] == 50.0
    assert row["exit_price"] == 55.0  # Çarşamba'nın son bilinen fiyatı
    assert row["realized_return"] == pytest.approx(0.10)


def test_performance_summary_computes_win_rate_and_cumulative_return():
    performance_tracker.log_completed_positions([
        {"entered_on": "2026-07-06", "exited_on": "2026-07-10", "ticker": "AAA",
         "entry_price": 100.0, "exit_price": 110.0, "realized_return": 0.10,
         "holding_days": 4, "exit_reason": "friday_liquidation"},
        {"entered_on": "2026-07-13", "exited_on": "2026-07-17", "ticker": "BBB",
         "entry_price": 50.0, "exit_price": 45.0, "realized_return": -0.10,
         "holding_days": 4, "exit_reason": "friday_liquidation"},
    ])

    summary = performance_tracker.summarize_performance()
    assert summary["n_positions"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["avg_return"] == pytest.approx(0.0)
    assert summary["cumulative_return"] == pytest.approx(1.10 * 0.90 - 1)
