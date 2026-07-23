"""
Tüm proje genelinde kullanılan sabitler ve konfigürasyon değerleri.

Ortam değişkenleri ile override edilebilir (.env dosyası, bkz. .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Dizinler ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Veri çekme ────────────────────────────────────────────────────────────
# Günlük mum verisi için ne kadar geriye gidileceği (indikatör ısınma payı
# dahil — MA200 gibi uzun pencereler için en az 250+ gün gerekir).
HISTORY_PERIOD = "2y"
HISTORY_INTERVAL = "1d"

# ── İndikatör parametreleri (varsayılanlar; genetik optimizer bunları
#    data/models/agent_params.json üzerinden override edebilir) ───────────
DEFAULT_PARAMS = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ma_short": 50,
    "ma_long": 200,
    "adx_period": 14,
    "adx_trend_threshold": 20,
    "atr_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "rel_volume_period": 20,
    "rel_volume_threshold": 1.5,
    # ── İkinci göz doğrulama katmanı (ConfirmationAgent) parametreleri ────
    "min_liquidity_try": 5_000_000.0,  # 20 günlük ort. min. TL cinsinden hacim
    "min_risk_reward": 1.5,            # min. kabul edilebilir R:R oranı
    "extreme_rsi_veto": 85,            # bu RSI seviyesi üstü AL'ı veto eder
    # ── Stop/Hedef hesaplama (trend gücüne göre değişken R:R) ─────────────
    # NOT: Eskiden stop=1.5xATR, hedef=3.0xATR sabitti -> R:R HER ZAMAN tam
    # olarak 2.0 çıkıyordu, hisseden hisseye hiç değişmiyordu. Bu da
    # ConfirmationAgent'ın "min_risk_reward" kontrolünü ayırt edici gücü
    # olmayan bir açma/kapama düğmesine indirgiyordu. Artık hedef, zaten
    # hesaplanan ADX (trend gücü) ile ölçekleniyor -- güçlü trendde hedef
    # daha uzağa konuyor, zayıf/yatay trendde daha yakın tutuluyor. Böylece
    # R:R gerçekten hisseden hisseye, günden güne değişiyor.
    "atr_stop_multiplier": 1.5,
    "atr_target_multiplier_base": 2.0,
    "atr_target_trend_bonus": 2.0,
    # ── Agent'lar arası görüş ayrılığı cezası ─────────────────────────────
    # NOT: has_conflict (agent'lar arası yön uyuşmazlığı) eskiden sadece
    # ekranda "⚠️" göstermek için hesaplanıyordu, final_score'u hiç
    # etkilemiyordu. Artık çelişki varsa skor bu çarpanla küçültülüyor.
    "conflict_penalty": 0.8,
}


def sanitize_params(params: dict | None) -> dict:
    """GA'nın (veya elle düzenlemenin) üretebileceği KENDİ KENDİNİ GEÇERSİZ
    KILAN bir parametre kombinasyonunu düzeltir.

    NOT (gerçek prodüksiyon hatası, tespit edilip düzeltildi): ATR bazlı
    stop/hedef formülü (bkz. supervisor.py::_compute_stop_target), verilen
    "atr_stop_multiplier", "atr_target_multiplier_base" ve
    "atr_target_trend_bonus" için ULAŞILABİLECEK EN YÜKSEK Risk/Ödül oranı
    şudur: (atr_target_multiplier_base + atr_target_trend_bonus) /
    atr_stop_multiplier (trend_strength=1.0, yani ADX>=50 olduğu en güçlü
    trend anında). "min_risk_reward" (ConfirmationAgent'ın eşiği) bu
    maksimumdan büyükse, HİÇBİR hisse HİÇBİR gün bu eşiği geçemez — yani
    AL sinyali üretilse bile ikinci onay katmanı onu HER ZAMAN reddeder.
    Bu, GA'nın PARAM_SEARCH_SPACE'inde bu dört değeri birbirinden bağımsız
    örneklemesi yüzünden gerçekten oluştu: 2026-07-18 tarihli GA çalıştırması
    tam olarak böyle bir kombinasyon üretti (max R:R≈1.29, min_risk_reward
    eşiği=2.93) ve o günden itibaren üretilen HER TEK "AL" sinyali (o gün
    watchlist'te 8 tane vardı) ikinci onaydan reddedildi. Bu fonksiyon,
    hem GA'nın yeni bireyler üretirken bu tuzağa düşmesini önler (bkz.
    genetic_optimizer.py) hem de data/models/agent_params.json içinde
    zaten kayıtlı olabilecek eski/bozuk bir kombinasyona karşı, parametreler
    her okunduğunda (run_daily_analysis.py) uygulanan bir son-savunma
    hattıdır. %15 pay bırakılır (sadece TEORİK olarak ulaşılabilir olması
    yetmez; pratikte de anlamlı bir şekilde geçilebilir olması istenir).
    """
    p = dict(params or {})
    stop_mult = float(p.get("atr_stop_multiplier", DEFAULT_PARAMS["atr_stop_multiplier"]))
    base = float(p.get("atr_target_multiplier_base", DEFAULT_PARAMS["atr_target_multiplier_base"]))
    bonus = float(p.get("atr_target_trend_bonus", DEFAULT_PARAMS["atr_target_trend_bonus"]))
    min_rr = float(p.get("min_risk_reward", DEFAULT_PARAMS["min_risk_reward"]))

    if stop_mult > 0:
        max_achievable_rr = (base + bonus) / stop_mult
        safe_cap = round(max_achievable_rr * 0.85, 2)
        if min_rr > safe_cap:
            p["min_risk_reward"] = max(0.5, safe_cap)
    return p


# ── Sinyal eşikleri ───────────────────────────────────────────────────────
# NOT (kalibrasyon): final_score = ağırlıklı ortalama(signal_value × confidence)
# formülünden geldiği için doğal olarak dar bir aralıkta toplanır — 5 agent'ın
# görüşleri birbirini kısmen iptal eder ve confidence çarpanı (ortalama ~0.35)
# skoru ek olarak küçültür. Sentetik veri üzerinde ölçülen gerçekçi dağılım:
# std≈0.044, p90≈0.07, gözlenen maksimum≈0.15. Eski eşik (0.35) bu aralığın
# ~8 katıydı ve pratikte hiçbir zaman aşılamıyordu — bu yüzden tüm sinyaller
# BEKLE'ye düşüyor, Stop/Hedef/R:R hep None çıkıyordu. Yeni eşikler, skor
# dağılımının üst ~%1-3'ünü (gerçek "güçlü konsensüs" günlerini) AL/SAT
# olarak işaretleyecek şekilde kalibre edildi. Gerçek BIST verisiyle
# dağılım farklılaşabilir; genetik optimizer periyodik olarak bu eşikleri
# de arama uzayına dahil edip zamanla iyileştirebilir.
SIGNAL_BUY_THRESHOLD = 0.12    # final_score bu değerin üstündeyse AL
SIGNAL_SELL_THRESHOLD = -0.12  # final_score bu değerin altındaysa SAT
# Aradaki bölge BEKLE olarak değerlendirilir.

# ── Agent başlangıç ağırlıkları (feedback loop bunları zamanla günceller) ──
DEFAULT_AGENT_WEIGHTS = {
    "trend_agent": 0.25,
    "rsi_agent": 0.20,
    "macd_agent": 0.20,
    "volume_agent": 0.15,
    "pattern_agent": 0.20,
}

# ── Backtest / işlem maliyeti ──────────────────────────────────────────
# Ortalama gidiş-dönüş işlem maliyeti (komisyon + BSMV + spread yaklaşık).
# Aracı kurumunuza göre değişir; GA ve backtest bu maliyeti düşerek NET
# getiriye göre optimize eder -- brüt (maliyetsiz) getiriye göre kalibre
# etmek gerçek karlılığı sistematik olarak abartabilir. Kendi aracı
# kurumunuzun oranına göre bu değeri güncelleyebilirsiniz.
DEFAULT_TRANSACTION_COST = 0.003  # %0.3 gidiş-dönüş (yaklaşık)

# ── Feedback / öğrenme ────────────────────────────────────────────────────
# Bir sinyalin "sonucu" kaç gün sonra değerlendirilir (1 haftalık swing
# trading ufkuna uygun).
EVALUATION_HORIZON_DAYS = 5

# Ağırlık güncelleme hızı (exponential moving update katsayısı)
WEIGHT_LEARNING_RATE = 0.05

# Genetik algoritma periyodu (kaç günde bir parametre optimizasyonu çalışır)
GENETIC_OPTIMIZATION_INTERVAL_DAYS = 7
# NOT (performans kalibrasyonu, 4. revizyon): Örneklem sayısı (ticker_sample)
# jenerasyon döngüsü BAŞLAMADAN ÖNCE bir kez seçilip o haftanın TÜM
# bireyleri/jenerasyonları için SABİT kalıyor (bkz. optimize_parameters).
# Bu yüzden küçük bir örneklem sadece gürültülü bir ölçüm değil, o haftaya
# özgü SABİT bir yanlılık demek -- popülasyon/jenerasyonu kısıp örneklemi
# küçük tutmak, GA'nın o yanlılığı daha hassas ezberlemesine (overfitting)
# yol açabilir. Bu yüzden örneklem BÜYÜK tutulup (50), popülasyon/
# jenerasyon TAM değerine (20/14) geri döndürüldü -- 90 dakikalık zaman
# aşımı keyfi bir güvenlik sınırıydı, gerçek bir kaynak kısıtı değildi;
# bu iş haftada bir kez çalıştığı için limit 150 dakikaya çıkarıldı
# (bkz. .github/workflows/nightly_analysis.yml). Tahmini gerçek süre
# ~97 dakika (~1.54x güvenlik payı, aylık CI bütçesinin sadece ~%21'i).
GENETIC_POPULATION_SIZE = 20
GENETIC_GENERATIONS = 14
GENETIC_WALKFORWARD_WINDOWS = 3
# GA, parametreleri TÜM watchlist yerine rastgele seçilmiş bu kadar
# hisse üzerinde değerlendirir (performans amaçlı) — bulunan parametreler
# yine de gece taramasında TÜM hisselere uygulanır, sadece GA'nın kendi
# iç değerlendirme maliyeti azalır. Her haftalık çalıştırmada farklı bir
# alt küme seçilir, böylece uzun vadede tüm watchlist örneklenmiş olur.
# 100 hisselik BIST100 evreninin yarısını her hafta örnekler.
GENETIC_TICKER_SAMPLE_SIZE = 50

# ── Dosya yolları ─────────────────────────────────────────────────────────
WEIGHTS_FILE = MODELS_DIR / "agent_weights.json"
PARAMS_FILE = MODELS_DIR / "agent_params.json"
PARAMS_HISTORY_FILE = PROCESSED_DATA_DIR / "params_history.csv"
PREDICTIONS_LOG_FILE = PROCESSED_DATA_DIR / "predictions_log.csv"
LATEST_SIGNALS_FILE = PROCESSED_DATA_DIR / "latest_signals.csv"
LATEST_SIGNALS_DETAIL_FILE = PROCESSED_DATA_DIR / "latest_signals_detail.json"
WEIGHT_HISTORY_FILE = PROCESSED_DATA_DIR / "weight_history.csv"

# ── Haftalık portföy döngüsü (Pazartesi-Cuma takip sistemi) ───────────────
# Sistem, AL sinyali verip ikinci onaydan (confirmed=True) geçen hisseleri
# haftalık bir döngüde takip eder: Pazartesi (ya da haftanın ilk işlem
# gününde) en iyi potansiyele sahip adaylar seçilir; Salı-Perşembe akşamları
# durumları gözden geçirilir (kötüleşen varsa ya da bariz daha güçlü bir
# alternatif varsa değiştirilmesi önerilir); Perşembe akşamı ayrıca ertesi
# gün (Cuma) kapanışta tüm pozisyonların satılacağı bildirilir; Cuma akşamı
# hem satış hatırlatması yapılır hem de gelecek haftanın (Pazartesi) aday
# listesi o günün taze verisiyle belirlenip kaydedilir. Bkz.
# src/portfolio/cycle_manager.py.
PORTFOLIO_MAX_HOLDINGS = 5        # aynı anda takip edilecek maksimum hisse sayısı
# Bir alternatifin, hâlâ AL/onaylı durumdaki en zayıf takip edilen hisseye
# göre "belirgin şekilde daha iyi potansiyele sahip" sayılması için gereken
# minimum final_score farkı (kötüleşme YOKSA bile isteğe bağlı yükseltme
# önerisi tetiklenir). final_score dağılımı dar olduğundan (bkz. yukarıdaki
# SIGNAL_BUY_THRESHOLD kalibrasyon notu, std≈0.044) 0.03 anlamlı bir eşiktir.
PORTFOLIO_UPGRADE_MARGIN = 0.03
PORTFOLIO_STATE_FILE = PROCESSED_DATA_DIR / "portfolio_state.json"
PORTFOLIO_CYCLE_REPORT_FILE = PROCESSED_DATA_DIR / "portfolio_cycle_report.json"
# Haftalık döngünün GERÇEKTEN önerdiği hisselerin (Pazartesi girişi -> Cuma
# çıkışı) simüle edilmiş getirisini kaydeder. GA'nın out-of-sample fitness'ı
# ve agent ağırlıkları hep sentetik backtest üzerinden hesaplanır; bu log
# ise "sistem gerçekten kâr ediyor mu" sorusunun tek somut kanıtıdır. Bkz.
# src/portfolio/performance_tracker.py.
CYCLE_PERFORMANCE_LOG_FILE = PROCESSED_DATA_DIR / "cycle_performance_log.csv"

# ── Bildirimler ───────────────────────────────────────────────────────────
# E-posta gönderimi GitHub Actions üzerindeki dawidd6/action-send-mail
# adımı tarafından yapılır; uygulama kodu SMTP kimlik bilgisi taşımaz.
# GitHub repo'unuzda şu üç Secret tanımlayın:
#   EMAIL_USERNAME     → gönderici Gmail adresi (örn. adiniz@gmail.com)
#   EMAIL_APP_PASSWORD → Gmail "Uygulama Şifresi" (Google Hesabı → Güvenlik)
#   EMAIL_TO           → alıcı adresi (örn. adiniz@gmail.com)
