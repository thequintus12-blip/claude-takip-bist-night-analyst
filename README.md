# BIST Night Analyst

Borsa İstanbul (BIST) hisseleri için **kısa vadeli swing trading** odaklı,
çoklu-agent teknik analiz raporlama sistemi.

> ⚠️ Bu proje **otomatik al-sat yapmaz**. Sadece AL / SAT / BEKLE raporu
> üretir; tüm işlem kararları kullanıcıya aittir. Yatırım tavsiyesi
> değildir.

## Ne Yapar?

Piyasa kapandıktan sonra (GitHub Actions ile, gece) çalışır:

1. Watchlist'teki 50 hisse için OHLCV verisini çeker (yfinance)
2. 5 uzman agent (Trend, RSI, MACD, Hacim, Formasyon) her hisseyi bağımsız
   değerlendirir
3. Bir supervisor bu görüşleri ağırlıklı olarak birleştirir, BIST100 rejim
   filtresi uygular, nihai AL/SAT/BEKLE kararını + ATR bazlı stop/hedef
   seviyelerini üretir
4. Geçmiş tahminlerin gerçekleşen getirisi değerlendirilir; agent'ların
   ağırlıkları buna göre güncellenir (kendi kendini geliştiren sistem)
5. Haftalık olarak genetik algoritma ile indikatör parametreleri walk-forward
   doğrulamayla optimize edilir
6. Sonuçlar repoya commit edilir; Streamlit arayüzü bu sonuçları gösterir

## Mimari

```
GitHub Actions (cron, piyasa kapanışı sonrası)
    → veri çek → indikatör hesapla → agent'lar çalışır → supervisor karar verir
    → sonuçlar data/processed/'a yazılır, repo'ya commit edilir
    → feedback loop: geçmiş tahminler değerlendirilir, ağırlıklar güncellenir
    → (haftalık) genetik optimizasyon: parametreler walk-forward ile iyileştirilir

Streamlit Cloud
    → sadece git'teki son sonuçları okur ve görselleştirir (hesaplama yapmaz)
```

## Look-Ahead Bias Koruması

Bu projenin en kritik tasarım ilkesi: **bir agent, kararını verdiği günden
sonraki hiçbir veriyi göremez.**

- `src/agents/base_agent.py::get_data_as_of()` — her agent çağrısından önce
  veriyi zorunlu olarak `as_of_date`'e kadar keser; agent kodu fiziksel
  olarak gelecek satırlara erişemez.
- `src/backtest/engine.py` — sinyal üretimi (`f(veri[0:t])`) ile sonuç
  ölçümü (`getiri[t+N]/fiyat[t] - 1`) her zaman ayrı, birbirine sızmayan
  hesaplamalardır.
- `src/learning/genetic_optimizer.py` — walk-forward validasyon: parametre
  optimizasyonu (fitness) sadece "train" penceresinde yapılır, "test"
  penceresi GA sürecine hiç gösterilmez, sadece out-of-sample raporlama
  için kullanılır.
- `src/learning/feedback_loop.py` — bir tahmin, `result_available_date`
  geçmeden değerlendirmeye katılmaz (henüz sonucu belli olmayan veri
  kullanılmaz).
- `scripts/audit_lookahead.py` — tüm tahmin logunu programatik olarak
  denetler, ihlal varsa CI'da hata verir.
- `tests/test_no_lookahead.py` — agent'a gelecekteki günler eklendiğinde
  geçmiş tarihli sinyalin DEĞİŞMEDİĞİNİ doğrulayan regresyon testi içerir.

## Kurulum

```bash
git clone <repo-url>
cd bist-night-analyst
pip install -r requirements.txt
cp .env.example .env   # opsiyonel: Telegram bildirimleri için
```

### Yerel test

```bash
python scripts/run_daily_analysis.py   # tek seferlik gece taraması
python scripts/audit_lookahead.py      # look-ahead denetimi
pytest tests/ -v                       # tüm testler
streamlit run streamlit_app.py         # arayüzü başlat
```

### GitHub Actions Kurulumu

1. Repoyu GitHub'a push edin.
2. (Opsiyonel) Settings → Secrets → Actions altına `TELEGRAM_BOT_TOKEN` ve
   `TELEGRAM_CHAT_ID` ekleyin.
3. `.github/workflows/nightly_analysis.yml` hafta içi her gün piyasa
   kapanışından sonra (18:30 TRT) otomatik çalışır, sonuçları commit eder.
4. Cumartesi günleri ayrı bir job genetik optimizasyonu çalıştırır.
5. Manuel tetiklemek için: Actions sekmesi → "BIST Night Analyst" →
   "Run workflow".

### Streamlit Cloud Deploy

1. [share.streamlit.io](https://share.streamlit.io) üzerinden repoyu bağlayın.
2. Ana dosya: `streamlit_app.py`
3. Python sürümü: 3.11 önerilir.
4. Ekstra sistem paketi gerekmez (TA-Lib kullanılmadığı için).

## Watchlist

`config/symbols.py` içinde 50 BIST hissesi tanımlıdır (SASA, THYAO, GARAN,
ASELS, TUPRS, vb.). Düzenlemek için bu dosyayı güncelleyin.

## Agent'lar

| Agent | Odak | Ana Sinyal Kaynağı |
|---|---|---|
| `trend_agent` | Genel trend yapısı | MA50/MA200, ADX |
| `rsi_agent` | Momentum / aşırı alım-satım | RSI(14) |
| `macd_agent` | Momentum dönüşü | MACD kesişimi, histogram |
| `volume_agent` | Kurumsal ilgi teyidi | Relative volume |
| `pattern_agent` | Kırılım kalitesi | BB squeeze, resistance breakout, mum kalitesi |

Her agent `-1.0` (güçlü sat) ile `+1.0` (güçlü al) arası bir `signal_value`,
bir `confidence` (0-1) ve insan-okunabilir bir `reasoning` üretir. Supervisor
bunları ağırlıklı ortalama + BIST100 rejim çarpanı ile birleştirir.

## İkinci Göz Doğrulama Katmanı (ConfirmationAgent)

5 agent + supervisor bir hisseye **AL** kararı verdikten SONRA devreye
giren, bağımsız bir doğrulama kapısı. Yeni bir oy eklemez — var olan kararı
üç ek kriterle bir kez daha süzer, hiçbir yeni indikatör hesaplamaz:

| Kriter | Ne kontrol eder | Varsayılan eşik |
|---|---|---|
| Likidite | 20 günlük ort. TL cinsinden işlem hacmi | ≥ 5.000.000 TL |
| Risk/Ödül | Supervisor'ın zaten hesapladığı ATR bazlı R:R | ≥ 1.5 |
| Aşırı RSI vetosu | Çok güçlü trend + çok yüksek RSI ("tükenme rallisi" riski) | RSI < 85 |

Bu üç eşik de haftalık genetik optimizasyonun arama uzayına dahildir
(`src/learning/genetic_optimizer.py::PARAM_SEARCH_SPACE`) — GA, backtest
sırasında AL sinyali doğrulanmazsa o günü BEKLE (işlem yapılmamış) olarak
sayar, böylece bu eşikler de zamanla gerçek performansa göre ayarlanır.

Streamlit'te bir hissenin detayına girildiğinde, AL sinyali varsa bu
katmanın "✅ Onaylandı" / "❌ Reddedildi" sonucu ve gerekçesi gösterilir.

## Sinyal Kalitesi İyileştirmeleri (Kod İncelemesi Sonucu)

Sistematik bir kod incelemesi sonucu, hiçbir yeni indikatör eklemeden
(mevcut "minimum indikatör" felsefesine sadık kalarak) tespit edilip
düzeltilen dört gerçek sorun:

1. **Risk/Ödül artık gerçekten değişken.** Eskiden stop=1.5×ATR,
   hedef=3.0×ATR sabitti — bu da R:R'nin HER hissede, HER gün tam olarak
   2.0 çıkması demekti. ConfirmationAgent'ın "min_risk_reward" kontrolü bu
   yüzden hiçbir ayırt edici güce sahip değildi. Artık hedef, zaten
   hesaplanan ADX (trend gücü) ile ölçekleniyor — güçlü trendde hedef
   uzağa, zayıf trendde yakına konuyor. Çarpanlar da GA ile ayarlanabilir.
2. **RSI artık aşırılık derecesini dikkate alıyor.** Eskiden RSI=71 ile
   RSI=95 aynı puanı alıyordu. Artık eşiğin ne kadar üstünde/altında
   olunduğuna göre kademeli puanlanıyor — tam da IEYHO örneğinde
   karşılaştığımız (RSI=87.5, güçlü trend) senaryoyu artık RSI agent'ının
   kendisi de "SAT" olarak işaretliyor, sadece ikinci göz doğrulamasına
   kalmıyor.
3. **Genetik optimizasyon artık işlem sayısını ve maksimum düşüşü de
   gözetiyor.** Eskiden fitness sadece ortalama getiriye bakıyordu; 1-2
   şanslı işlemden çıkan yüksek ama kırılgan sonuçlar "en iyi" seçilebiliyordu.
   `backtest/metrics.py`'nin zaten hesapladığı ama kullanılmayan
   `trade_count` ve `max_drawdown` artık fitness'a dahil.
4. **Agent'lar arası görüş ayrılığı artık kararı gerçekten etkiliyor.**
   `has_conflict` bayrağı eskiden sadece arayüzde "⚠️" göstermek içindi;
   final skoru hiç etkilemiyordu. Artık çelişki varsa skor bir ceza
   çarpanıyla (varsayılan 0.8, GA ile ayarlanabilir) küçültülüyor.
5. **(Bonus) Ağırlık güncellemesi az örnekle aşırı tepki vermiyor.**
   Sistem yeni çalıştırıldığında bir agent'ın 2-3 değerlendirilmiş tahmini
   olabilir; ham ortalama kullanılsaydı tek bir tahmin accuracy'yi
   %0'dan %100'e sıçratabilirdi. Artık Bayesian yumuşatma ile az veri
   varken %50 (nötr) civarında tutuluyor, veri arttıkça gerçek orana
   yakınsıyor.

### İkinci Tur: Kalan Sert Eşiklerin Kademelileştirilmesi

Aynı "ikili değil kademeli" prensibi, ilk turda henüz dokunulmamış üç yere
daha uygulandı:

6. **TrendAgent'ın "zayıf trend" dalları artık ADX'e göre kademeli.**
   Eskiden ADX=0.1 ile ADX=19.9 (eşiğe çok yakın) aynı sabit puanı
   alıyordu. Artık eşiğe yakınlık oranında puan/güven artıyor.
7. **BIST100 rejim çarpanı artık ADX=20'de sıçramıyor.** Eskiden
   ADX=19.9 → 1.0x, ADX=20.1 → 1.15x gibi ani, yapay bir sıçrama vardı.
   Artık trend gücüyle sürekli (kademeli) ölçekleniyor.
8. **Çelişki cezası artık görüş ayrılığının şiddetiyle orantılı.**
   Eskiden hafif bir çelişki (+0.15/-0.15) ile şiddetli bir çelişki
   (+0.9/-0.9) aynı sabit cezayı alıyordu. Artık zıt görüşler ne kadar
   birbirinden uzaksa ceza o kadar büyük.

### Üçüncü Tur: Prodüksiyonda Fiilen Gözlenen İki Hata

Bu iki madde, önceki turlardaki gibi teorik bir inceleme değil; repo'da
**gerçekten kayıtlı veride gözlenen** iki hatanın tespiti ve düzeltilmesidir.

9. **İkinci onay katmanı, GA'nın bulduğu bir parametre kombinasyonuyla
   kalıcı olarak kilitlenebiliyordu.** ATR bazlı stop/hedef formülünün
   üretebileceği maksimum Risk/Ödül oranı `(atr_target_multiplier_base +
   atr_target_trend_bonus) / atr_stop_multiplier` iledir. GA, bu üç
   çarpanı `min_risk_reward` eşiğinden tamamen bağımsız aradığı için,
   2026-07-18 tarihli optimizasyon çalıştırması eşiği (2.93), o
   kombinasyonun ulaşabileceği MAKSİMUM R:R'nin (≈1.29) üzerinde seçti —
   yani hiçbir hisse, hiçbir gün bu eşiği geçemez hale geldi. Nitekim
   2026-07-21 taramasında AL sinyali veren 8 hissenin 8'i de sadece bu
   yüzden reddedildi. `config/settings.py::sanitize_params()` artık her
   birey üretiminde/mutasyonunda/crossover'ında (GA tarafında) VE her
   parametre dosyası okunduğunda (`run_daily_analysis.py` tarafında, son
   savunma hattı olarak) bu eşiği her zaman ulaşılabilir bir üst sınıra
   kelepçeliyor.
10. **Canlı geri bildirim döngüsü, GA'nın optimize ettiğinden farklı bir
    ufku ölçüyordu.** Bir tahminin sonucu `as_of_date + 5 TAKVİM günü`
    sonra değerlendiriliyordu; ama backtest/GA her zaman tam 5 İŞLEM
    günü (satır) ileri bakıyor. Hafta sonu araya girdiğinde gerçek ufuk
    3-4 işlem gününe düşüyor, yani agent ağırlıkları GA'nın hedeflediğinden
    daha kısa bir pencereye göre güncelleniyordu. Artık takvim günü
    yerine iş günü (`pandas.tseries.offsets.BDay`) kullanılıyor.

## Haftalık Portföy Döngüsü

`src/portfolio/cycle_manager.py`, AL sinyali verip ikinci onaydan
(`confirmed=True`) geçen hisseleri haftalık bir döngüde takip eder. Bu
bölüm sadece **yönlendirme** sunar — otomatik alım/satım yapılmaz, karar
her zaman kullanıcıya aittir. Akış:

| Gün | Aksiyon |
|---|---|
| Haftanın ilk işlem günü (normalde Pazartesi) | Bir önceki Cuma belirlenen adaylar takibe alınır (yoksa o günün taze analizinden seçilir); aynı akşam bu adayların durumu hemen bir kez daha gözden geçirilir. |
| Salı - Perşembe | Takip edilen hisselerin durumu o günün taze verisiyle yeniden değerlendirilir: AL/onay durumunu kaybeden "kötüleşti" sayılır ve yerine takip-dışı en güçlü aday önerilir; kötüleşen yoksa bile belirgin şekilde daha güçlü bir alternatif varsa "yükseltme" önerilir. |
| Perşembe | Yukarıdaki gözden geçirmeye ek olarak, ertesi gün (Cuma) kapanışta hangi hisselerin satılacağı önceden bildirilir. |
| Cuma | Bugün kapanışta tüm takip edilen hisselerin satılması gerektiği bildirilir; takip listesi boşaltılır; o günün taze analiziyle gelecek haftanın (Pazartesi) aday listesi belirlenip kaydedilir — BIST hafta sonu işlem görmediği için "hafta sonu analizi" fiilen Cuma'nın kendi kapanış-sonrası taramasıdır. |

Durum `data/processed/portfolio_state.json`'da (takip edilenler + gelecek
hafta adayları), gösterilecek rapor ise `data/processed/portfolio_cycle_report.json`'da
saklanır; ikisi de her gece `run_daily_analysis.py` tarafından güncellenir
ve Streamlit'in **📅 Haftalık Portföy Döngüsü** bölümünde, e-posta
bildiriminde ve log'larda gösterilir.

Ayarlanabilir parametreler (`config/settings.py`):

- `PORTFOLIO_MAX_HOLDINGS` (varsayılan 5) — aynı anda takip edilecek maksimum hisse sayısı.
- `PORTFOLIO_UPGRADE_MARGIN` (varsayılan 0.03) — kötüleşme olmasa bile bir "yükseltme" önerisi tetiklemek için gereken minimum final_score farkı.

Bilinen sınırlama: sistem BIST resmi tatil takvimini bilmiyor;
Perşembe/Cuma bildirimleri o günün analiz tarihinin haftanın kaçıncı
günü olduğuna bakılarak tetiklenir (bkz. modül docstring'i). "Yeni hafta
başlangıcı" tespiti buna karşı dayanıklıdır (Pazartesi tatilse haftanın
ilk çalıştığı gün doğru şekilde algılanır).

## Sistem Kendini Nasıl Geliştiriyor?

Sistemde üç ayrı, birbirini tamamlayan öğrenme/kendini-düzeltme mekanizması
vardır. Üçü de **otomatik alım/satım yapmaz** — sadece sinyal kalitesini
ve parametreleri zamanla iyileştirir.

| Mekanizma | Ne zaman çalışır | Neyi öğrenir | Dosya |
|---|---|---|---|
| Agent ağırlık geri beslemesi | Her gece | Hangi agent'ın yön tahmini daha isabetli, ona göre `agent_weights.json`'daki ağırlığı artırır/azaltır | `src/learning/feedback_loop.py` |
| Genetik parametre optimizasyonu | Haftalık (Cumartesi) | RSI/MACD/ADX eşikleri, ATR çarpanları, ikinci onay eşikleri — walk-forward, look-ahead'siz | `src/learning/genetic_optimizer.py` |
| Haftalık döngü gerçekleşen performansı | Her Cuma kapanışı | Sistemin GERÇEKTEN önerdiği hisse sepetinin, Pazartesi-Cuma tutulduğunda ne kazandırdığı | `src/portfolio/performance_tracker.py` |

### Genetik optimizasyon artık iki şekilde "daha verimli" sinyale odaklı

1. **Fitness fonksiyonu artık getiri volatilitesini de cezalandırıyor.**
   `backtest/metrics.py::summarize()` zaten `return_std` (işlemler arası
   getiri standart sapması) hesaplıyordu ama fitness bunu hiç
   kullanmıyordu — aynı ortalama getiriye sahip, biri istikrarlı diğeri
   çok oynak iki parametre seti GA için birebir eşdeğerdi. Artık
   `_robust_fitness_score`, Sharpe oranının temel mantığıyla (getiri ÷
   risk) aynı şekilde, getiri dalgalanmasını da doğrudan cezalandırıyor
   — GA artık "yüksek ama kırılgan" değil, **istikrarlı ve öngörülebilir**
   sinyal üreten parametreleri tercih ediyor.
2. **"Ratchet" koruması: sistem artık asla geriye gitmiyor.** Eskiden
   `optimize_parameters()`, o haftanın GA sonucu ne olursa olsun
   `agent_params.json`'ı koşulsuz üzerine yazıyordu — kötü/şansına
   overfit olmuş bir hafta, önceki haftaların daha iyi parametrelerini
   sessizce silebiliyordu. Artık her hafta bulunan yeni aday, **canlıda
   olan mevcut parametrelerle aynı test pencerelerinde** yeniden
   değerlendiriliyor; yeni aday kanıtlanmış şekilde daha iyi değilse
   dosyaya hiç dokunulmuyor. `data/processed/params_history.csv`'ye artık
   her deneme (benimsense de reddedilse de) bir `adopted` bayrağıyla
   kaydediliyor, böylece Streamlit'te "GA kaç haftada bir gerçekten bir
   şey buluyor" görülebiliyor.

### Haftalık döngünün gerçek (simüle edilmiş) performans takibi

GA'nın out-of-sample fitness'ı ve agent ağırlıkları hep **sentetik**,
tek-tek sinyal bazlı bir backtest üzerinden hesaplanır ("bu sinyalden N
gün sonra ne oldu"). Ama kullanıcının fiilen takip ettiği şey bu değil —
kullanıcı her hafta sistemin seçtiği hisse **sepetini** Pazartesi'den
Cuma'ya kadar tutuyor (bkz. Haftalık Portföy Döngüsü). Bu iki şey her
zaman aynı sonucu vermez (örn. hafta ortasında kötüleşen bir hisse
değiştirilmemişse Cuma'ya kadar sepette kalır — sentetik backtest bunu
modellemez).

`src/portfolio/performance_tracker.py`, her Cuma kapanışta gerçekten
tamamlanan her pozisyonun (giriş fiyatı → çıkış fiyatı) getirisini
`data/processed/cycle_performance_log.csv`'ye kaydeder ve Streamlit'in
**📈 Haftalık Döngünün Gerçekleşen Performansı** bölümünde kümülatif
getiri, kazanma oranı ve tüm geçmiş pozisyonlar olarak gösterir. Bu,
sistemin gerçekten kâr edip etmediğinin — yani "kendini geliştirme"
iddiasının — tek somut, denetlenebilir kanıtıdır.

**Doğal bir sonraki adım** (bu turda kapsam dışı bırakıldı, kasıtlı):
bu gerçekleşen performans verisi biriktikçe, GA'nın ratchet karşılaştırmasına
veya agent ağırlıklarına ek bir girdi olarak bağlanabilir. Şimdilik
sadece görünürlük/doğrulama amacıyla tutuluyor — yeterli veri (en az
birkaç ay, çeşitli piyasa koşulları) birikmeden bunu otomatik bir karara
bağlamak, az sayıda haftanın gürültüsüne aşırı uyum riski taşır.

## Neden RL/LangGraph/TA-Lib/VectorBT Kullanılmadı?

Orijinal mimaride önerilen bu araçlar bilinçli olarak sadeleştirildi:

- **TA-Lib** → Streamlit Cloud'da C derlemesi gerektirir, kırılgan. Tüm
  indikatörler sıfırdan, sadece pandas/numpy ile yazıldı.
- **Stable-Baselines3/PPO** → Streamlit Cloud'un ücretsiz katmanında RL
  eğitimi pratik değil. `src/learning/` altında iskelet olarak bırakıldı
  (`environment.py`, `ppo_trainer.py`), aktif kullanılmıyor. Öğrenme,
  skor-bazlı ağırlık güncelleme + genetik algoritma ile yapılıyor.
- **LangGraph** → Agent'lar arası gerçek bir diyalog/döngü olmadığı (hepsi
  paralel çalışıp tek seferde birleştiriliyor) için gereksiz bağımlılık.
  `src/agents/supervisor.py` saf Python ile aynı işi yapıyor.
- **VectorBT** → Lisans/kurulum karmaşıklığı; `src/backtest/` altında
  kendi vektörize backtest motoru yazıldı.

## Dizin Yapısı

```
bist-night-analyst/
├── config/              # Watchlist ve tüm konfigürasyon
├── data/
│   ├── raw/              # Ham OHLCV (parquet)
│   ├── processed/        # Sinyal sonuçları, tahmin logu, ağırlık geçmişi,
│   │                      # portföy döngüsü durumu (portfolio_state.json,
│   │                      # portfolio_cycle_report.json), gerçekleşen
│   │                      # haftalık performans (cycle_performance_log.csv)
│   └── models/           # agent_weights.json, agent_params.json
├── src/
│   ├── data_collector/   # yfinance veri çekme + temizleme
│   ├── indicators/       # Sıfırdan indikatör hesaplama
│   ├── agents/            # 5 uzman agent + supervisor + look-ahead koruması
│   ├── learning/          # Feedback loop + genetik optimizer (ratchet korumalı) (+ RL iskelet)
│   ├── backtest/          # Look-ahead güvenli backtest + metrikler
│   ├── portfolio/         # Haftalık portföy döngüsü (cycle_manager.py) +
│   │                      # gerçekleşen performans takibi (performance_tracker.py)
│   └── reporter/          # Rapor üretimi + Telegram bildirimi
├── scripts/               # run_daily_analysis.py, train_model.py, audit_lookahead.py
├── tests/                 # Look-ahead, indikatör, veri ve döngü testleri
├── .github/workflows/     # Gece + haftalık GitHub Actions
└── streamlit_app.py       # Dashboard (sadece okuma, hesaplama yapmaz)
```
