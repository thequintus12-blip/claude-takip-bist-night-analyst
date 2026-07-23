"""
Takip edilecek BIST hisseleri.

yfinance üzerinden veri çekebilmek için tüm BIST sembollerine ".IS" suffix'i
eklenir. BIST100 endeksi rejim filtresi için ayrıca tutulur.
"""

from __future__ import annotations

# Ham BIST ticker kodları (suffix'siz)
# NOT: Bu liste, likit-10 haftalık rotasyon yerine SABİT BIST100 evrenine
# genişletildi (bkz. proje geçmişi). Rotasyonun "yetim tahmin" riski
# (bir hisse listeden çıkınca sonucu hiç değerlendirilemeyen bekleyen
# tahminler) ve azalan istatistiksel derinlik nedeniyle sabit, geniş bir
# evren tercih edildi. Likidite filtresi zaten ConfirmationAgent'ta var;
# "en likit N'i göster" ihtiyacı Streamlit'te bir GÖRÜNTÜLEME filtresi
# olarak karşılanıyor (bkz. streamlit_app.py), watchlist'in kendisi
# değişmiyor.
WATCHLIST: list[str] = [
    "AEFES", "AKBNK", "AKSA", "AKSEN", "ALARK", "ALTNY", "ANSGR", "ARCLK",
    "ASELS", "ASTOR", "BALSU", "BERA", "BIMAS", "BRSAN", "BRYAT", "BSOKE",
    "BTCIM", "CANTE", "CCOLA", "CIMSA", "CVKMD", "CWENE", "DAPGM", "DOAS",
    "DOHOL", "DSTKF", "ECILC", "EFOR", "EKGYO", "ENERY", "ENJSA", "ENKAI",
    "EREGL", "ESEN", "EUPWR", "EUREN", "FENER", "FROTO", "GARAN", "GENIL",
    "GESAN", "GLRMK", "GRSEL", "GRTHO", "GSRAY", "GUBRF", "HALKB", "HEKTS",
    "IEYHO", "ISCTR", "ISMEN", "IZENR", "KCHOL", "KLRHO", "KRDMD", "KTLEV",
    "KUYAS", "MAGEN", "MAVI", "MGROS", "MIATK", "MPARK", "OBAMS", "ODAS",
    "ODINE", "OTKAR", "OYAKC", "PAHOL", "PASEU", "PATEK", "PETKM", "PGSUS",
    "PSGYO", "QUAGR", "RALYH", "REEDR", "SAHOL", "SARKY", "SASA", "SISE",
    "SKBNK", "SOKM", "TAVHL", "TCELL", "THYAO", "TKFEN", "TOASO", "TRALT",
    "TRENJ", "TRMET", "TSKB", "TTKOM", "TUKAS", "TUPRS", "TURSG", "ULKER",
    "VAKBN", "VESTL", "YKBNK", "ZOREN",
]

# Piyasa rejim filtresi için referans endeks
BENCHMARK_SYMBOL: str = "XU100"


def to_yfinance_ticker(symbol: str) -> str:
    """BIST sembolünü yfinance formatına çevirir (örn. 'SASA' -> 'SASA.IS')."""
    symbol = symbol.strip().upper()
    if symbol == BENCHMARK_SYMBOL:
        return "XU100.IS"
    if symbol.endswith(".IS"):
        return symbol
    return f"{symbol}.IS"


def get_yfinance_watchlist() -> list[str]:
    """Tüm watchlist'i yfinance ticker formatında döndürür."""
    return [to_yfinance_ticker(s) for s in WATCHLIST]


def get_benchmark_ticker() -> str:
    return to_yfinance_ticker(BENCHMARK_SYMBOL)
