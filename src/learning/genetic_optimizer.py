"""
Aşama 2 öğrenme mekanizması: walk-forward genetik algoritma ile indikatör
parametre optimizasyonu.

KRİTİK: Fitness hesaplaması SADECE train penceresinde yapılır. Test
penceresi, bulunan en iyi parametrenin gerçek (out-of-sample) performansını
ölçmek için kullanılır ve hiçbir genetik operasyon (seçim, mutasyon,
crossover) test penceresinin verisine erişemez. Bu ayrım fonksiyon
imzalarında da fiziksel olarak ayrı DataFrame'ler verilerek garanti edilir.
"""

from __future__ import annotations

import json
import logging
import random
from copy import deepcopy

import pandas as pd

from config.settings import (
    DEFAULT_PARAMS, EVALUATION_HORIZON_DAYS, GENETIC_GENERATIONS,
    GENETIC_POPULATION_SIZE, GENETIC_TICKER_SAMPLE_SIZE,
    GENETIC_WALKFORWARD_WINDOWS, PARAMS_FILE, PARAMS_HISTORY_FILE,
    sanitize_params,
)
from src.agents.confirmation_agent import ConfirmationAgent
from src.agents.supervisor import analyze_watchlist, build_agents
from src.backtest.engine import simulate_signals, walk_forward_splits
from src.backtest.metrics import summarize
from src.indicators.feature_engineer import build_features

logger = logging.getLogger(__name__)

# Optimize edilecek parametreler ve arama aralıkları (min, max, tip)
PARAM_SEARCH_SPACE = {
    "rsi_period": (10, 21, int),
    "rsi_oversold": (20, 35, int),
    "rsi_overbought": (65, 80, int),
    "macd_fast": (8, 15, int),
    "macd_slow": (20, 30, int),
    "adx_trend_threshold": (15, 30, int),
    "rel_volume_threshold": (1.2, 2.5, float),
    "bb_std": (1.5, 2.5, float),
    # ── İkinci göz doğrulama katmanı (ConfirmationAgent) ──────────────────
    "min_liquidity_try": (1_000_000, 15_000_000, float),
    "min_risk_reward": (1.0, 3.0, float),
    "extreme_rsi_veto": (75, 92, int),
    # ── Stop/hedef ve çelişki cezası (bkz. supervisor.py) ─────────────────
    "atr_stop_multiplier": (1.0, 2.5, float),
    "atr_target_multiplier_base": (1.5, 3.0, float),
    "atr_target_trend_bonus": (1.0, 4.0, float),
    "conflict_penalty": (0.5, 1.0, float),
}


# NOT (düzeltilen hata): "min_risk_reward" ve ATR çarpanları (atr_stop_multiplier,
# atr_target_multiplier_base, atr_target_trend_bonus) burada birbirinden
# TAMAMEN bağımsız örneklenir/mutasyona uğrar/crossover'lanır. Bu dört değer
# arasında hiçbir ilişki gözetilmezse, GA'nın kendisi -- gerçekten olduğu gibi
# (bkz. 2026-07-18 tarihli data/models/agent_params.json) -- ulaşılması
# TEORİK OLARAK İMKANSIZ bir min_risk_reward eşiği üretebilir (örn. ATR
# çarpanlarının üretebileceği maksimum R:R 1.29 iken eşik 2.93 seçilmiş, bu
# yüzden hiçbir AL sinyali asla onaylanamamıştı). sanitize_params() her
# bireyin üretildiği/değiştiği üç noktada da (rastgele üretim, mutasyon,
# crossover) çağrılarak bu tuzağın GA arama sürecine tekrar girmesini önler.
def _random_individual() -> dict:
    individual = dict(DEFAULT_PARAMS)
    for key, (lo, hi, typ) in PARAM_SEARCH_SPACE.items():
        if typ is int:
            individual[key] = random.randint(lo, hi)
        else:
            individual[key] = round(random.uniform(lo, hi), 2)
    return sanitize_params(individual)


def _mutate(individual: dict, rate: float = 0.25) -> dict:
    child = deepcopy(individual)
    for key, (lo, hi, typ) in PARAM_SEARCH_SPACE.items():
        if random.random() < rate:
            if typ is int:
                child[key] = random.randint(lo, hi)
            else:
                child[key] = round(random.uniform(lo, hi), 2)
    return sanitize_params(child)


def _crossover(parent_a: dict, parent_b: dict) -> dict:
    child = {}
    for key in PARAM_SEARCH_SPACE:
        child[key] = parent_a[key] if random.random() < 0.5 else parent_b[key]
    return sanitize_params({**DEFAULT_PARAMS, **child})


def _sample_tickers_for_ga(
    raw_data: dict[str, pd.DataFrame], benchmark_ticker: str, sample_size: int,
) -> dict[str, pd.DataFrame]:
    """GA'nın her (birey × jenerasyon × pencere) kombinasyonu için yeniden
    işlediği hisse sayısını sınırlar (performans). Benchmark her zaman
    dahil edilir; geri kalanından rastgele bir alt küme seçilir. GA
    SHARED (tüm watchlist için ortak) parametreleri optimize ettiğinden,
    temsili bir alt küme yeterli bir fitness tahmini verir — bulunan
    parametreler yine de gece taramasında TAM watchlist'e uygulanır, bu
    örnekleme sadece GA'nın kendi iç değerlendirme maliyetini azaltır.
    Her çalıştırmada farklı bir alt küme seçilerek, uzun vadede (haftalık
    tekrarlarla) tüm watchlist'in davranışı örneklenmiş olur."""
    non_benchmark = [t for t in raw_data if t != benchmark_ticker]
    if len(non_benchmark) <= sample_size:
        return raw_data
    sampled = random.sample(non_benchmark, sample_size)
    result = {t: raw_data[t] for t in sampled}
    if benchmark_ticker in raw_data:
        result[benchmark_ticker] = raw_data[benchmark_ticker]
    return result


def _build_features_for_individual(
    raw_data: dict[str, pd.DataFrame], benchmark_ticker: str, params: dict,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Belirli bir birey (parametre seti) için TÜM hisselerin feature'larını
    BİR KEZ hesaplar. build_features() sonucu sadece (df, params) çiftine
    bağlıdır, pencere sınırlarına bağlı DEĞİLDİR — bu yüzden aynı bireyin
    farklı pencerelerde (train/test, N adet walk-forward penceresi) tekrar
    tekrar build_features() çağırması tamamen gereksizdi. Bu, GA'da tespit
    edilen ikinci büyük performans darboğazıydı (n_windows kat gereksiz
    tekrar). Bu fonksiyon sonucu, aynı birey için tüm pencerelerde yeniden
    kullanılır."""
    benchmark_df = raw_data.get(benchmark_ticker)
    benchmark_features = build_features(benchmark_df, None, params) if benchmark_df is not None else None
    features_by_ticker = {
        ticker: build_features(df, benchmark_df["Close"] if benchmark_df is not None else None, params)
        for ticker, df in raw_data.items() if ticker != benchmark_ticker
    }
    return features_by_ticker, benchmark_features


# Bu kadar az işlemden çıkan sonuç istatistiksel olarak güvenilmez sayılır
MIN_RELIABLE_TRADES = 3
# İki parametre seti aynı ortalama getiriye sahipse, işlemler arası getirisi
# daha DÜZENSİZ (yüksek standart sapmalı) olan, birim risk başına daha az
# verim üretiyor demektir. Bu katsayı, drawdown_penalty'nin (0.3) katsayısıyla
# aynı "getiri birimi" mantığıyla kalibre edilmiştir — ikisi de doğrudan
# avg_ret'ten çıkarılan, aynı ölçekteki cezalardır.
VOLATILITY_PENALTY_WEIGHT = 0.5


def _robust_fitness_score(metrics: dict) -> float | None:
    """Tek bir (hisse, pencere) sonucundan, sadece ortalama getiriye değil
    işlem sayısına (güvenilirlik), maksimum düşüşe (risk) ve getirinin ne
    kadar TUTARLI olduğuna (verimlilik) da bakan bir fitness skoru üretir.

    NOT (kalibrasyon — ilk tur): Eskiden fitness SADECE avg_return'dü.
    backtest/metrics.py::summarize() zaten trade_count ve max_drawdown'ı
    hesaplıyordu ama bunlar hiç kullanılmıyordu. Bu, GA'nın 1-2 şanslı
    işlemden çıkan yüksek ama kırılgan sonuçları "en iyi" parametre seti
    olarak seçmesine karşı hiçbir koruma sağlamıyordu (overfitting
    riski). Az işlem sayısı güven çarpanıyla, büyük maksimum düşüş de
    getiriden düşülerek cezalandırılmaya başlanmıştı.

    NOT (kalibrasyon — ikinci tur, "verimlilik"): summarize() ayrıca
    return_std (işlemler arası getiri standart sapması) ve sharpe (Sharpe
    oranı) hesaplıyordu, ama fitness bunları da hiç kullanmıyordu. Yani
    aynı ortalama getiriye sahip, biri İSTİKRARLI diğeri ÇOK OYNAK iki
    parametre seti GA için birebir eşdeğerdi — sistem "yüksek getiri"yi
    "verimli/tutarlı getiri"den ayırt edemiyordu. Artık getiri
    dalgalanması da (Sharpe oranının temel mantığıyla aynı şekilde: getiri
    ÷ risk) doğrudan cezalandırılıyor, böylece GA daha DÜZENLİ ve
    öngörülebilir sinyaller üreten parametreleri tercih ediyor.
    """
    avg_ret = metrics.get("avg_return")
    n_trades = metrics.get("trade_count", 0)
    max_dd = metrics.get("max_drawdown")
    ret_std = metrics.get("return_std")

    if pd.isna(avg_ret) or n_trades == 0:
        return None

    reliability = min(1.0, n_trades / MIN_RELIABLE_TRADES)
    drawdown_penalty = 0.3 * abs(max_dd) if pd.notna(max_dd) and max_dd < 0 else 0.0
    volatility_penalty = VOLATILITY_PENALTY_WEIGHT * ret_std if pd.notna(ret_std) else 0.0

    return (avg_ret - drawdown_penalty - volatility_penalty) * reliability


def _fitness_on_precomputed_features(
    features_by_ticker: dict[str, pd.DataFrame], benchmark_features: pd.DataFrame | None,
    params: dict, window_start: pd.Timestamp, window_end: pd.Timestamp,
) -> float:
    """_fitness_on_window ile aynı mantık, ancak feature'lar ÖNCEDEN
    hesaplanmış olarak alınır (bkz. _build_features_for_individual).
    Bu fonksiyona window dışına ait hiçbir veri sızdırılmaz: feature'lar
    tüm geçmişle hesaplanmış olsa da sinyal/skor sadece pencere içindeki
    tarihler için değerlendirilir.
    """
    weights = {name: 1.0 / 5 for name in build_agents().keys()}
    all_scores = []
    confirmation_agent = ConfirmationAgent(params)

    for ticker, features in features_by_ticker.items():
        window_dates = features.loc[window_start:window_end].index
        if len(window_dates) == 0:
            continue

        signals_in_window = {}
        agents = build_agents(params)
        for date in window_dates:
            try:
                from src.agents.supervisor import analyze_ticker
                result = analyze_ticker(ticker, features, date, weights, agents, benchmark_features, params)
                final_signal = result["final_signal"]
                # AL sinyali, ikinci göz doğrulamasından geçmezse fitness
                # hesabında BEKLE (işlem yapılmamış) olarak sayılır — bu
                # sayede GA, confirmation eşiklerinin (likidite/R:R/RSI
                # vetosu) gerçek getiriyi nasıl etkilediğini "görebilir".
                if final_signal == "AL":
                    conf_result = confirmation_agent.review(ticker, features, date, result)
                    if not conf_result.confirmed:
                        final_signal = "BEKLE"
                signals_in_window[date] = final_signal
            except Exception:  # noqa: BLE001
                continue

        if not signals_in_window:
            continue

        signal_series = pd.Series(signals_in_window)
        sim = simulate_signals(signal_series, features["Close"].reindex(signal_series.index),
                                 horizon=EVALUATION_HORIZON_DAYS)
        metrics = summarize(sim)
        score = _robust_fitness_score(metrics)
        if score is not None:
            all_scores.append(score)

    if not all_scores:
        return -1.0
    return sum(all_scores) / len(all_scores)


def optimize_parameters(
    raw_data: dict[str, pd.DataFrame], benchmark_ticker: str,
    population_size: int = None, generations: int = None,
    n_windows: int = None, ticker_sample_size: int = None,
) -> dict:
    """Walk-forward GA ile en iyi parametre setini bulur ve out-of-sample
    (test penceresi) performansını raporlar."""
    population_size = population_size or GENETIC_POPULATION_SIZE
    generations = generations or GENETIC_GENERATIONS
    n_windows = n_windows or GENETIC_WALKFORWARD_WINDOWS
    ticker_sample_size = ticker_sample_size or GENETIC_TICKER_SAMPLE_SIZE

    original_ticker_count = len([t for t in raw_data if t != benchmark_ticker])
    raw_data = _sample_tickers_for_ga(raw_data, benchmark_ticker, ticker_sample_size)
    sampled_count = len([t for t in raw_data if t != benchmark_ticker])
    if sampled_count < original_ticker_count:
        logger.info(
            "Performans amaçlı: %d hisseden %d tanesi GA değerlendirmesi için "
            "örneklendi (bulunan parametreler yine de TÜM watchlist'e uygulanır).",
            original_ticker_count, sampled_count,
        )

    sample_ticker = next(iter(raw_data))
    all_dates = pd.DatetimeIndex(raw_data[sample_ticker].index)
    windows = walk_forward_splits(all_dates, n_windows=n_windows)

    if not windows:
        logger.warning("Walk-forward pencereleri oluşturulamadı, varsayılan parametreler kullanılacak.")
        return dict(DEFAULT_PARAMS)

    population = [_random_individual() for _ in range(population_size)]
    best_individual, best_fitness = None, float("-inf")

    for generation in range(generations):
        scored = []
        for individual in population:
            # Bu birey için feature'lar BİR KEZ hesaplanır, tüm pencerelerde
            # yeniden kullanılır (bkz. _build_features_for_individual).
            features_by_ticker, benchmark_features = _build_features_for_individual(
                raw_data, benchmark_ticker, individual
            )
            # Fitness, SADECE train pencerelerinin ortalaması üzerinden
            # hesaplanır — test pencereleri bu aşamada hiç görülmez.
            train_scores = [
                _fitness_on_precomputed_features(
                    features_by_ticker, benchmark_features, individual, w.train_start, w.train_end
                )
                for w in windows
            ]
            fitness = sum(train_scores) / len(train_scores) if train_scores else -1.0
            scored.append((fitness, individual))

        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_fitness:
            best_fitness, best_individual = scored[0]

        logger.info("Jenerasyon %d/%d — en iyi train fitness: %.4f", generation + 1, generations, scored[0][0])

        # Elitizm: en iyi %25 hayatta kalır, geri kalan crossover+mutasyon ile üretilir
        survivors = [ind for _, ind in scored[: max(2, population_size // 4)]]
        new_population = list(survivors)
        while len(new_population) < population_size:
            parent_a, parent_b = random.sample(survivors, 2)
            child = _crossover(parent_a, parent_b)
            child = _mutate(child)
            new_population.append(child)
        population = new_population

    # Out-of-sample doğrulama: en iyi bireyin TEST pencerelerindeki
    # performansı — bu, GA sürecine hiçbir şekilde geri beslenmez, sadece
    # raporlama amaçlıdır.
    best_features_by_ticker, best_benchmark_features = _build_features_for_individual(
        raw_data, benchmark_ticker, best_individual
    )
    test_scores = [
        _fitness_on_precomputed_features(
            best_features_by_ticker, best_benchmark_features, best_individual, w.test_start, w.test_end
        )
        for w in windows
    ]
    oos_performance = sum(test_scores) / len(test_scores) if test_scores else float("nan")

    # ── Regresyon koruması (ratchet) ──────────────────────────────────────
    # NOT (self-improvement iyileştirmesi): Bu fonksiyon eskiden, o haftanın
    # GA sonucu ne olursa olsun (şansına göre train verisine aşırı uyum
    # sağlamış, out-of-sample'da ZAYIF bir birey bile olsa) agent_params.json'ı
    # KOŞULSUZ üzerine yazıyordu. Yani sistem "kendini geliştirmek" yerine
    # her hafta rastgele yön değiştirebiliyordu — kötü bir hafta, önceki
    # haftaların daha iyi parametrelerini sessizce silebiliyordu. Artık: bu
    # haftanın adayı, HÂLİHAZIRDA CANLI OLAN parametrelerle AYNI test
    # pencerelerinde (adil bir karşılaştırma için) yeniden değerlendirilir;
    # yeni aday gerçekten daha iyi değilse dosyaya HİÇ dokunulmaz — sistem
    # sadece kanıtlanmış bir iyileşme olduğunda ileri gider, asla geriye
    # gitmez.
    current_params = None
    if PARAMS_FILE.exists():
        try:
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                current_params = sanitize_params(json.load(f).get("params", {}))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Mevcut agent_params.json okunamadı, ilk çalıştırma gibi davranılacak: %s", exc)

    if current_params is None:
        current_oos = float("-inf")
        logger.info("Kayıtlı bir parametre seti yok (ilk çalıştırma) — yeni sonuç doğrudan benimsenecek.")
    else:
        current_features_by_ticker, current_benchmark_features = _build_features_for_individual(
            raw_data, benchmark_ticker, current_params
        )
        current_test_scores = [
            _fitness_on_precomputed_features(
                current_features_by_ticker, current_benchmark_features, current_params, w.test_start, w.test_end
            )
            for w in windows
        ]
        current_oos = sum(current_test_scores) / len(current_test_scores) if current_test_scores else float("-inf")

    adopted = pd.notna(oos_performance) and oos_performance > current_oos
    logger.info(
        "Aday out-of-sample=%.4f | Mevcut (canlı) out-of-sample=%.4f | %s",
        oos_performance if pd.notna(oos_performance) else float("nan"), current_oos,
        "BENİMSENDİ (geliştirme kanıtlandı)" if adopted else "REDDEDİLDİ (mevcuttan daha iyi değil)",
    )

    result = {
        "params": best_individual if adopted else (current_params or dict(DEFAULT_PARAMS)),
        "train_fitness": best_fitness,
        "out_of_sample_performance": oos_performance,
        "current_oos_performance": None if current_oos == float("-inf") else current_oos,
        "adopted": adopted,
        "optimized_at": str(pd.Timestamp.today().date()),
    }

    if adopted:
        with open(PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        logger.info(
            "agent_params.json değiştirilmedi — bu haftanın GA sonucu, "
            "canlıdaki parametrelerden daha iyi bir out-of-sample performans "
            "göstermedi."
        )

    _append_params_history(result)

    return result


def _append_params_history(result: dict) -> None:
    """Her haftalık GA çalıştırmasının train_fitness/out_of_sample_performance
    değerlerini ve bulunan parametreleri params_history.csv'ye ekler — GA
    bir iyileşme BULAMASA (adopted=False) bile bu çalıştırma kaydedilir,
    böylece "GA kaç haftada bir gerçekten daha iyi bir şey buluyor"
    sorusu da izlenebilir.

    NOT (düzeltilen hata): PARAMS_HISTORY_FILE sabiti settings.py'de tanımlıydı
    ve streamlit_app.py bu dosyayı OKUYUP grafiğe çiziyordu, ama bu fonksiyon
    hiç yazılmamıştı — yani "GA Kalibrasyon Geçmişi" grafiği hiçbir zaman
    dolamıyordu, agent_params.json her hafta üzerine yazılırken geçmiş veri
    hiç birikmiyordu. Bu, GA'nın gerçekten zamanla iyileşip iyileşmediğini
    izlemenin TEK yoludur; bu fonksiyon olmadan o soru asla cevaplanamaz."""
    row = {
        "date": pd.Timestamp.today().normalize(),
        "train_fitness": result.get("train_fitness"),
        "out_of_sample_performance": result.get("out_of_sample_performance"),
        "current_oos_performance": result.get("current_oos_performance"),
        "adopted": result.get("adopted", True),  # eski satırlar (bu alan eklenmeden önce) her zaman benimsenmişti
        **{f"param_{k}": v for k, v in result.get("params", {}).items()},
    }
    if PARAMS_HISTORY_FILE.exists():
        hist = pd.read_csv(PARAMS_HISTORY_FILE, parse_dates=["date"])
        if "adopted" not in hist.columns:
            hist["adopted"] = True  # bu sütun eklenmeden önceki tüm haftalar koşulsuz benimseniyordu
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(PARAMS_HISTORY_FILE, index=False)
