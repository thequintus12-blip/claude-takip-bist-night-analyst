"""
Aşama 1 öğrenme mekanizması: skor-bazlı ağırlık güncelleme.

Kural: Bir tahminin sonucu, sadece result_available_date geçmişse
değerlendirmeye katılır. Bu, gün t'de henüz kapanmamış (sonucu belli
olmayan) bir tahminin o günün feedback hesaplamasına sızmasını engeller —
feedback_loop'un kendi içindeki look-ahead koruması budur.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


def load_weights() -> dict[str, float]:
    if settings.WEIGHTS_FILE.exists():
        with open(settings.WEIGHTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return dict(settings.DEFAULT_AGENT_WEIGHTS)


def save_weights(weights: dict[str, float]) -> None:
    with open(settings.WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)


def load_predictions_log() -> pd.DataFrame:
    if settings.PREDICTIONS_LOG_FILE.exists():
        return pd.read_csv(settings.PREDICTIONS_LOG_FILE, parse_dates=[
            "as_of_date", "result_available_date"
        ])
    return pd.DataFrame(columns=[
        "as_of_date", "ticker", "agent", "signal", "signal_value", "confidence",
        "weight", "result_available_date", "actual_return", "was_correct", "evaluated",
    ])


def save_predictions_log(df: pd.DataFrame) -> None:
    df.to_csv(settings.PREDICTIONS_LOG_FILE, index=False)


def append_predictions(new_rows: pd.DataFrame) -> None:
    """Yeni üretilen (henüz değerlendirilmemiş) tahminleri log'a ekler."""
    log = load_predictions_log()
    combined = pd.concat([log, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["as_of_date", "ticker", "agent"], keep="last")
    save_predictions_log(combined)


def evaluate_due_predictions(price_lookup: dict[str, pd.Series], today: pd.Timestamp) -> pd.DataFrame:
    """result_available_date <= today olan ve henüz evaluated=False olan
    satırları, gerçekleşen fiyat verisiyle değerlendirir.

    price_lookup: {ticker: Close fiyat serisi (tüm geçmiş, bugüne kadar)}
    """
    log = load_predictions_log()
    if log.empty:
        return log

    # dtype'ları gevşet (object) ki sonradan bool/float değer atarken
    # pandas'ın katı dtype kontrolü (LossySetitemError) hata vermesin.
    for col in ("actual_return", "was_correct", "evaluated"):
        if col in log.columns:
            log[col] = log[col].astype(object)

    due_mask = (~log["evaluated"].astype(bool)) & (log["result_available_date"] <= today)
    due = log[due_mask]

    for idx, row in due.iterrows():
        ticker = row["ticker"]
        if ticker not in price_lookup:
            continue
        prices = price_lookup[ticker]

        as_of = row["as_of_date"]
        result_date = row["result_available_date"]
        prices_until_result = prices.loc[:result_date]
        if as_of not in prices.index or prices_until_result.empty:
            continue

        entry_price = prices.loc[as_of]
        exit_price = prices_until_result.iloc[-1]
        actual_return = (exit_price / entry_price) - 1

        predicted_direction = 1 if row["signal"] == "AL" else (-1 if row["signal"] == "SAT" else 0)
        actual_direction = 1 if actual_return > 0 else (-1 if actual_return < 0 else 0)
        was_correct = predicted_direction != 0 and predicted_direction == actual_direction

        log.loc[idx, "actual_return"] = actual_return
        log.loc[idx, "was_correct"] = was_correct
        log.loc[idx, "evaluated"] = True

    save_predictions_log(log)
    return log


def update_weights_from_feedback(learning_rate: float = None) -> dict[str, float]:
    """Değerlendirilmiş (evaluated=True) tahminlerin doğruluk oranına göre
    her agent'ın ağırlığını exponential moving update ile günceller.

    weight_new = weight_old * (1 - lr) + accuracy * lr

    NOT (kalibrasyon): accuracy, ham ortalama (mean) yerine Bayesian
    yumuşatma (Beta-benzeri önsel) ile hesaplanıyor. Sistem yeni
    çalıştırıldığında bir agent'ın sadece 2-3 değerlendirilmiş tahmini
    olabilir -- ham ortalama kullanılırsa tek bir şanslı/şanssız tahmin
    accuracy'yi %0'dan %100'e sıçratabilir, bu da ağırlığı gürültüye göre
    savurur. Yumuşatma, az örnekle %50 (nötr) civarında tutar, örnek
    sayısı arttıkça gerçek gözlemlenen orana yaklaşır.
    """
    learning_rate = learning_rate or settings.WEIGHT_LEARNING_RATE
    log = load_predictions_log()
    weights = load_weights()

    evaluated = log[log["evaluated"].astype(bool) & log["signal"].isin(["AL", "SAT"])]
    if evaluated.empty:
        logger.info("Henüz değerlendirilmiş tahmin yok, ağırlıklar değişmedi.")
        return weights

    prior_strength = 10  # sanal "önsel" örnek sayısı -- az veri varken bunun etkisi baskın
    prior_accuracy = 0.5

    for agent_name in weights:
        agent_rows = evaluated[evaluated["agent"] == agent_name]
        if agent_rows.empty:
            continue
        n = len(agent_rows)
        correct = agent_rows["was_correct"].astype(bool).sum()
        accuracy = (correct + prior_strength * prior_accuracy) / (n + prior_strength)
        old_weight = weights[agent_name]
        weights[agent_name] = round(old_weight * (1 - learning_rate) + accuracy * learning_rate, 4)
        logger.info(
            "%s: accuracy=%.3f (yumuşatılmış, n=%d ham=%.3f), weight %.4f -> %.4f",
            agent_name, accuracy, n, correct / n, old_weight, weights[agent_name],
        )

    # Negatif/sıfır ağırlığa düşmesini engelle, normalize et
    weights = {k: max(0.05, v) for k, v in weights.items()}
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    save_weights(weights)
    _append_weight_history(weights)
    return weights


def _append_weight_history(weights: dict[str, float]) -> None:
    today = pd.Timestamp.today().normalize()
    row = {"date": today, **weights}
    if settings.WEIGHT_HISTORY_FILE.exists():
        hist = pd.read_csv(settings.WEIGHT_HISTORY_FILE, parse_dates=["date"])
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(settings.WEIGHT_HISTORY_FILE, index=False)
