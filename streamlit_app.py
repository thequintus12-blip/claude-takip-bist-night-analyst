"""
BIST Night Analyst — Streamlit Dashboard

Bu arayüz HİÇBİR ağır hesaplama yapmaz; sadece GitHub Actions'ın gece
ürettiği ve repo'ya commit ettiği sonuçları (data/processed/*.csv) okur ve
görselleştirir. "Manuel Tara" bölümü sadece test/geliştirme amaçlıdır ve
GitHub Actions akışının yerine geçmez.

Bu uygulama OTOMATİK AL-SAT YAPMAZ. Sadece analiz/rapor üretir; tüm işlem
kararları kullanıcıya aittir.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from config.symbols import WATCHLIST
from src.learning.feedback_loop import load_predictions_log, load_weights
from src.portfolio import performance_tracker
from src.reporter.report_generator import agent_breakdown_dataframe, load_full_results

st.set_page_config(page_title="BIST Night Analyst", page_icon="📊", layout="wide")

st.title("📊 BIST Night Analyst")
st.caption(
    "Kısa vadeli swing trading için çoklu-agent teknik analiz raporu. "
    "Otomatik işlem yapılmaz — yalnızca AL / SAT / BEKLE önerisi sunar."
)

st.warning(
    "⚠️ Bu uygulama yatırım tavsiyesi değildir. Tüm sinyaller geçmiş veri ve "
    "teknik göstergelere dayalı otomatik analizdir; nihai karar kullanıcıya aittir.",
    icon="⚠️",
)


# ── Veri yükleme ────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def load_latest_signals() -> pd.DataFrame:
    if settings.LATEST_SIGNALS_FILE.exists():
        return pd.read_csv(settings.LATEST_SIGNALS_FILE)
    return pd.DataFrame()


@st.cache_data(ttl=900)
def load_weight_history() -> pd.DataFrame:
    if settings.WEIGHT_HISTORY_FILE.exists():
        return pd.read_csv(settings.WEIGHT_HISTORY_FILE, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data(ttl=900)
def load_predictions() -> pd.DataFrame:
    return load_predictions_log()


@st.cache_data(ttl=900)
def load_signal_details() -> dict:
    return load_full_results(settings.LATEST_SIGNALS_DETAIL_FILE)


@st.cache_data(ttl=900)
def load_params_history() -> pd.DataFrame:
    if settings.PARAMS_HISTORY_FILE.exists():
        return pd.read_csv(settings.PARAMS_HISTORY_FILE, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data(ttl=900)
def load_cycle_report() -> dict | None:
    if settings.PORTFOLIO_CYCLE_REPORT_FILE.exists():
        import json
        with open(settings.PORTFOLIO_CYCLE_REPORT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


signals_df = load_latest_signals()
weight_history_df = load_weight_history()
predictions_df = load_predictions()
current_weights = load_weights()
detail_by_ticker = load_signal_details()
params_history_df = load_params_history()
cycle_report = load_cycle_report()


# ── Değerlendirme durumu (kaç sinyal sonuçlandı?) ──────────────────────
# Bu bölüm bilinçli olarak en üstte: "kaç AL sinyali sonuçlandı, kaçı
# hâlâ bekliyor" sorusunun cevabını aramadan, sayfayı açar açmaz görmek
# için. Az sayıda sonuçlanmış sinyalle performans yorumu yapmanın
# istatistiksel olarak güvenilir olmadığını hatırlatır.
if not predictions_df.empty:
    tradeable = predictions_df[predictions_df["signal"].isin(["AL", "SAT"])]
    unique_predictions = tradeable.drop_duplicates(subset=["as_of_date", "ticker"])
    n_total = len(unique_predictions)
    n_evaluated = int(unique_predictions["evaluated"].astype(bool).sum())
    n_pending = n_total - n_evaluated

    st.info(
        f"📋 **Değerlendirme Durumu:** Şimdiye kadar {n_total} AL/SAT sinyali üretildi — "
        f"{n_evaluated} tanesi sonuçlandı ({settings.EVALUATION_HORIZON_DAYS} iş günü geçti), "
        f"{n_pending} tanesi hâlâ bekliyor. "
        + ("Bu sayı çok düşükken performans hakkında kesin yorum yapmak güvenilir değildir; "
           "istatistiksel olarak anlamlı bir değerlendirme için genellikle en az birkaç "
           "düzine sonuçlanmış sinyal beklenir." if n_evaluated < 15 else ""),
        icon="📋",
    )


# ── Haftalık portföy döngüsü ────────────────────────────────────────────
# AL sinyali verip ikinci onaydan geçen hisseler haftalık bir döngüde takip
# edilir: hafta başında seçilir, hafta içi akşamları gözden geçirilir
# (kötüleşen varsa değiştirilmesi önerilir), Cuma kapanışta tamamı satılır
# ve o günün analiziyle gelecek haftanın adayları belirlenir. Bkz.
# src/portfolio/cycle_manager.py. Bu bölüm sadece YÖNLENDİRME sunar —
# otomatik alım/satım yapılmaz.
st.subheader("📅 Haftalık Portföy Döngüsü")
st.caption(
    "AL sinyali verip ikinci onaydan geçen hisseler haftalık bir döngüde "
    "takip edilir: Pazartesi seçilir, hafta içi akşamları gözden geçirilir, "
    "Cuma kapanışta tamamı satılır, hafta sonu analiziyle gelecek haftanın "
    "adayları belirlenir. Otomatik alım/satım yapılmaz — sadece yönlendirme sunar."
)

if cycle_report is None:
    st.info(
        "Henüz bir haftalık döngü raporu üretilmemiş. İlk gece taramasından "
        "sonra burada görünecek."
    )
else:
    st.caption(f"Son güncelleme: {cycle_report['as_of_date']} — {cycle_report['day_type_label']}")

    if cycle_report["is_friday_liquidation"]:
        if cycle_report["liquidation_list"]:
            st.error(
                "🔴 **Bugün borsa kapanışında satın:** " + ", ".join(cycle_report["liquidation_list"]),
                icon="🔴",
            )
        if cycle_report["next_week_candidates"]:
            st.info(
                "📋 **Gelecek hafta (Pazartesi) için adaylar:** "
                + ", ".join(cycle_report["next_week_candidates"])
            )
    elif cycle_report["is_thursday_notice"] and cycle_report["liquidation_list"]:
        st.warning(
            "⏰ **Yarın (Cuma) kapanışta satılacak:** " + ", ".join(cycle_report["liquidation_list"]),
            icon="⏰",
        )

    if cycle_report["held_after"]:
        st.write("**📌 Şu an takip edilen hisseler:**")
        held_cols = st.columns(len(cycle_report["held_after"]))
        for col, ticker in zip(held_cols, cycle_report["held_after"]):
            col.metric(ticker, "AL ✅")
    else:
        st.info("Şu an takip edilen hisse yok.")

    if cycle_report["new_entries"]:
        st.success("🆕 Bu çalıştırmada yeni takibe alınanlar: " + ", ".join(cycle_report["new_entries"]))

    for w in cycle_report["worsened"]:
        st.warning(f"⚠️ **{w['ticker']}**: {w['reason']}")

    for s in cycle_report["swap_suggestions"]:
        verb = "durumu kötüleştiği için" if s["reason"] == "worsened" else "daha güçlü bir alternatif bulunduğu için"
        sell_score = f"{s['sell_score']:.3f}" if s["sell_score"] is not None else "?"
        st.info(
            f"🔁 **Öneri:** {s['sell_ticker']} yerine {s['buy_ticker']} değerlendirilebilir "
            f"({verb}; skor {sell_score} → {s['buy_score']:.3f})."
        )

    for m in cycle_report["messages"]:
        st.caption(f"ℹ️ {m}")


# ── Üst özet ────────────────────────────────────────────────────────────
if signals_df.empty:
    st.info(
        "Henüz bir gece taraması çalıştırılmamış. GitHub Actions iş akışı "
        "(`.github/workflows/nightly_analysis.yml`) ilk çalıştığında sonuçlar "
        "burada görünecek. Manuel test için aşağıdaki bölümü kullanabilirsiniz."
    )
else:
    col1, col2, col3, col4 = st.columns(4)
    n_buy = (signals_df["Sinyal"] == "AL").sum()
    n_sell = (signals_df["Sinyal"] == "SAT").sum()
    n_wait = (signals_df["Sinyal"] == "BEKLE").sum()
    last_date = signals_df["Tarih"].iloc[0] if "Tarih" in signals_df.columns else "-"
    col1.metric("🟢 AL Sinyali", int(n_buy))
    col2.metric("🔴 SAT Sinyali", int(n_sell))
    col3.metric("🟡 BEKLE", int(n_wait))
    col4.metric("Son Tarama", last_date)

    st.subheader("Güncel Tarama Sonuçları")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        signal_filter = st.multiselect("Sinyale göre filtrele", ["AL", "SAT", "BEKLE"], default=["AL", "SAT"])
    with filter_col2:
        liquidity_options = {"Tümünü göster": None, "En likit 10": 10, "En likit 25": 25, "En likit 50": 50}
        liquidity_choice = st.selectbox(
            "Likidite filtresi (sadece görüntüleme — watchlist değişmez)",
            list(liquidity_options.keys()),
        )
        top_n = liquidity_options[liquidity_choice]

    filtered = signals_df[signals_df["Sinyal"].isin(signal_filter)] if signal_filter else signals_df

    if top_n is not None and "Likidite (TL)" in filtered.columns:
        filtered = filtered.sort_values("Likidite (TL)", ascending=False).head(top_n)
        st.caption(
            f"Bugünün en likit {top_n} hissesi gösteriliyor (20 günlük ort. TL "
            "hacme göre). Bu sadece bir görüntüleme filtresidir; taranan "
            "watchlist etkilenmez."
        )

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.subheader("Hisse Bazında Agent Detayı")
    selected_ticker = st.selectbox("Hisse seçin", sorted(signals_df["Hisse"].unique()))
    st.caption(
        f"{selected_ticker} için her agent'ın bireysel görüşü ve gerekçesi "
        "aşağıda listelenir. Bu şeffaflık, neden o sinyalin üretildiğini "
        "anlamanızı sağlar."
    )

    detail = detail_by_ticker.get(selected_ticker)
    if detail is None:
        st.warning(
            f"{selected_ticker} için detaylı agent verisi bulunamadı. Bu genellikle "
            "gece taramasının bu güncellemeden ÖNCE çalıştırılmış olmasından "
            "kaynaklanır — bir sonraki tarama sonrası burası otomatik dolacaktır.",
            icon="⚠️",
        )
    else:
        summary_cols = st.columns(4)
        summary_cols[0].metric("Nihai Sinyal", detail.get("final_signal", "-"))
        summary_cols[1].metric("Skor", f"{detail.get('final_score', 0):.3f}")
        summary_cols[2].metric("Rejim Çarpanı", f"{detail.get('regime_multiplier', 1):.2f}")
        summary_cols[3].metric("Görüş Ayrılığı", "Evet ⚠️" if detail.get("has_conflict") else "Hayır")
        if detail.get("regime_note"):
            st.caption(detail["regime_note"])

        st.dataframe(agent_breakdown_dataframe(detail), use_container_width=True, hide_index=True)

        if detail.get("final_signal") == "AL":
            st.divider()
            st.subheader("🔎 İkinci Göz Doğrulaması")
            st.caption(
                "Birincil sistem AL kararı verdikten sonra, bağımsız üç kriterle "
                "(likidite, risk/ödül, aşırı RSI vetosu) bir kez daha süzülür."
            )
            confirmed = detail.get("confirmed")
            notes = detail.get("confirmation_notes", "")
            checks = detail.get("confirmation_checks", {})

            if confirmed is True:
                st.success(f"✅ Doğrulandı — {notes}")
            elif confirmed is False:
                st.error(f"❌ Doğrulanmadı — {notes}")
            else:
                st.info("Doğrulama verisi mevcut değil.")

            if checks:
                check_labels = {"likidite": "Likidite", "risk_odul": "Risk/Ödül", "asiri_rsi": "RSI Vetosu"}
                check_cols = st.columns(len(checks))
                for col, (key, passed) in zip(check_cols, checks.items()):
                    col.metric(check_labels.get(key, key), "✅" if passed else "❌")


# ── Agent ağırlık geçmişi ───────────────────────────────────────────────
st.subheader("🧠 Agent Ağırlıklarının Zaman İçindeki Gelişimi")
st.caption(
    "Her agent'ın geçmiş tahmin başarısına göre güncellenen güven ağırlığı. "
    "Yükselen bir çizgi, o agent'ın son dönemde daha isabetli olduğu anlamına gelir."
)

if weight_history_df.empty:
    st.info("Henüz ağırlık geçmişi yok — ilk feedback döngüsünden sonra burada birikecek.")
else:
    fig = go.Figure()
    for col in weight_history_df.columns:
        if col == "date":
            continue
        fig.add_trace(go.Scatter(x=weight_history_df["date"], y=weight_history_df[col],
                                  mode="lines+markers", name=col))
    fig.update_layout(template="plotly_dark", height=400, yaxis_title="Ağırlık", xaxis_title="Tarih")
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Güncel Ağırlıklar:**")
    weight_cols = st.columns(len(current_weights))
    for col, (agent, w) in zip(weight_cols, current_weights.items()):
        col.metric(agent.replace("_agent", "").upper(), f"{w:.2%}")


# ── Genetik optimizasyon geçmişi ────────────────────────────────────────
st.subheader("🧬 Haftalık Genetik Optimizasyon Geçmişi")
st.caption(
    "Her hafta bulunan parametrelerin train (içeride) ve out-of-sample "
    "(hiç görülmemiş veride) performansı. Out-of-sample çizgisi zamanla "
    "yükseliyorsa optimizasyon gerçekten işe yarıyor demektir; rastgele "
    "yukarı-aşağı sıçrıyorsa mevcut strateji setinin güçlü bir kenarı "
    "olmayabilir — bu durumda parametre ayarından çok stratejinin "
    "kendisini gözden geçirmek gerekebilir."
)

if params_history_df.empty:
    st.info(
        "Henüz GA geçmişi yok — ilk haftalık optimizasyon çalıştıktan sonra "
        "burada birikmeye başlayacak."
    )
else:
    if "adopted" in params_history_df.columns:
        n_adopted = int(params_history_df["adopted"].fillna(True).astype(bool).sum())
        st.caption(
            f"Şimdiye kadar {len(params_history_df)} haftalık denemeden **{n_adopted} tanesi** "
            "canlıya alındı (geri kalanı, mevcut parametrelerden daha iyi bir out-of-sample "
            "performans göstermediği için reddedildi — sistem sadece kanıtlanmış iyileşmelerde "
            "ilerler, asla geriye gitmez)."
        )

    fig_ga = go.Figure()
    fig_ga.add_trace(go.Scatter(
        x=params_history_df["date"], y=params_history_df["train_fitness"],
        mode="lines+markers", name="Train Fitness",
    ))
    if "adopted" in params_history_df.columns:
        adopted_mask = params_history_df["adopted"].fillna(True).astype(bool)
        fig_ga.add_trace(go.Scatter(
            x=params_history_df.loc[adopted_mask, "date"],
            y=params_history_df.loc[adopted_mask, "out_of_sample_performance"],
            mode="markers", name="Out-of-Sample (✅ Benimsendi)",
            marker=dict(symbol="circle", size=11, color="#2ecc71"),
        ))
        fig_ga.add_trace(go.Scatter(
            x=params_history_df.loc[~adopted_mask, "date"],
            y=params_history_df.loc[~adopted_mask, "out_of_sample_performance"],
            mode="markers", name="Out-of-Sample (⛔ Reddedildi)",
            marker=dict(symbol="x", size=9, color="#e74c3c"),
        ))
    else:
        fig_ga.add_trace(go.Scatter(
            x=params_history_df["date"], y=params_history_df["out_of_sample_performance"],
            mode="lines+markers", name="Out-of-Sample Performans",
        ))
    fig_ga.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_ga.update_layout(template="plotly_dark", height=350, yaxis_title="Fitness Skoru", xaxis_title="Tarih")
    st.plotly_chart(fig_ga, use_container_width=True)

    with st.expander("Haftalık bulunan parametrelerin tamamını görüntüle"):
        st.dataframe(params_history_df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)


# ── Haftalık döngünün GERÇEKLEŞEN performansı ───────────────────────────
# NOT: yukarıdaki GA out-of-sample fitness'ı ve agent ağırlıkları hep
# SENTETİK, tek-tek sinyal bazlı bir backtest üzerinden hesaplanır. Bu
# bölüm ise haftalık döngünün (bkz. cycle_manager.py) GERÇEKTEN önerdiği
# hisselerin, Pazartesi'den Cuma'ya tutulduğu varsayımıyla, ne
# kazandırdığını gösterir — sistemin kendini geliştirip geliştirmediğinin
# tek somut, denetlenebilir kanıtı budur.
st.subheader("📈 Haftalık Döngünün Gerçekleşen Performansı")
st.caption(
    "Yukarıdaki GA grafiği sentetik bir backteste dayanır; bu bölüm ise "
    "haftalık döngünün (📅 bölümü) GERÇEKTEN önerdiği hisselerin, Pazartesi "
    "girişinden Cuma çıkışına kadar tutulduğu varsayımıyla ne kazandırdığını "
    "gösterir."
)

perf_log = performance_tracker.load_performance_log()
perf_summary = performance_tracker.summarize_performance(perf_log)

if perf_summary["n_positions"] == 0:
    st.info("Henüz tamamlanmış bir haftalık döngü yok — ilk Cuma kapanışından sonra burada birikmeye başlayacak.")
else:
    perf_cols = st.columns(4)
    perf_cols[0].metric("Tamamlanan Pozisyon", perf_summary["n_positions"])
    perf_cols[1].metric("Kazanma Oranı", f"{perf_summary['win_rate']:.1%}")
    perf_cols[2].metric("Ortalama Getiri", f"{perf_summary['avg_return']:+.2%}")
    perf_cols[3].metric("Kümülatif Getiri", f"{perf_summary['cumulative_return']:+.2%}")

    priced = perf_log.dropna(subset=["realized_return"]).sort_values("exited_on")
    if not priced.empty:
        cumulative_curve = (1 + priced["realized_return"]).cumprod() - 1
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=priced["exited_on"], y=cumulative_curve, mode="lines+markers", name="Kümülatif Getiri",
        ))
        fig_perf.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_perf.update_layout(
            template="plotly_dark", height=300, yaxis_title="Kümülatif Getiri",
            xaxis_title="Çıkış Tarihi", yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with st.expander("Tamamlanan tüm pozisyonları görüntüle"):
        st.dataframe(perf_log.sort_values("exited_on", ascending=False), use_container_width=True, hide_index=True)


# ── Geçmiş tahmin doğruluğu ─────────────────────────────────────────────
st.subheader("📈 Geçmiş Tahmin Performansı")

if predictions_df.empty:
    st.info("Henüz değerlendirilmiş tahmin yok.")
else:
    evaluated = predictions_df[predictions_df["evaluated"].astype(bool, errors="ignore") == True]  # noqa: E712
    if evaluated.empty:
        st.info(
            f"Tahminler henüz değerlendirme aşamasında "
            f"({settings.EVALUATION_HORIZON_DAYS if hasattr(settings, 'EVALUATION_HORIZON_DAYS') else 5} "
            "gün sonra sonuçlanır)."
        )
    else:
        agent_accuracy = (
            evaluated.groupby("agent")["was_correct"]
            .apply(lambda s: s.astype(bool).mean())
            .sort_values(ascending=False)
        )
        fig2 = go.Figure(go.Bar(
            x=agent_accuracy.index, y=agent_accuracy.values,
            marker_color="#22c55e", text=[f"{v:.1%}" for v in agent_accuracy.values],
            textposition="outside",
        ))
        fig2.update_layout(
            template="plotly_dark", height=350, yaxis_title="Doğruluk Oranı",
            yaxis_tickformat=".0%", xaxis_title="Agent",
        )
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Ham tahmin geçmişini görüntüle"):
            st.dataframe(evaluated.sort_values("as_of_date", ascending=False), use_container_width=True)


# ── Manuel test taraması (opsiyonel, geliştirme amaçlı) ────────────────
with st.expander("🔧 Manuel Tarama Çalıştır (test amaçlı, gece otomasyonunun yerine geçmez)"):
    st.caption(
        "Bu buton, GitHub Actions akışını beklemeden tek seferlik bir tarama "
        "çalıştırır. Sonuçlar dosyaya kaydedilmez, sadece burada gösterilir."
    )
    selected_subset = st.multiselect("Test edilecek hisseler", WATCHLIST, default=WATCHLIST[:5])

    if st.button("Şimdi Tara", type="primary"):
        with st.spinner("Veri çekiliyor ve analiz ediliyor..."):
            from config.symbols import get_benchmark_ticker, to_yfinance_ticker
            from src.agents.supervisor import analyze_watchlist
            from src.data_collector.collector import download_batch
            from src.data_collector.preprocessor import clean_watchlist_data
            from src.indicators.feature_engineer import build_features, build_features_for_watchlist

            tickers = [to_yfinance_ticker(s) for s in selected_subset]
            benchmark_ticker = get_benchmark_ticker()
            raw = clean_watchlist_data(download_batch(tickers + [benchmark_ticker]))

            if not raw:
                st.error("Veri çekilemedi. Streamlit Cloud'da ağ/erişim sorunu olabilir.")
            else:
                features = build_features_for_watchlist(raw, benchmark_ticker)
                benchmark_features = (
                    build_features(raw[benchmark_ticker]) if benchmark_ticker in raw else None
                )
                as_of = min(df.index.max() for df in features.values())
                results = analyze_watchlist(features, as_of, current_weights, {}, benchmark_features)

                from src.reporter.report_generator import results_to_dataframe
                st.dataframe(results_to_dataframe(results), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Mimari notu: tüm sinyaller, üretildikleri tarihe kadar olan veriyle "
    "hesaplanır (look-ahead bias korumalı). Detaylar için README.md."
)
