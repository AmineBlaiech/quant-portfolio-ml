"""
Liste de 100 actions du S&P 500 utilisées comme univers d'investissement.
Choisies parmi les plus grandes capitalisations, réparties sur différents secteurs.
"""

TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AVGO", "ORCL", "CRM", "ADBE",
    "AMD", "CSCO", "ACN", "INTC", "IBM", "QCOM", "TXN", "INTU", "AMAT", "NOW",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "CMG",
    # Communication Services
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C",
    "SCHW", "BLK", "SPGI", "PGR", "CB",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "MDT", "GILD", "CVS", "ISRG",
    # Industrials
    "GE", "CAT", "RTX", "HON", "UNP", "BA", "DE", "LMT", "UPS", "ADP",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "PM", "MDLZ", "CL",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Materials
    "LIN", "SHW",
    # Utilities
    "NEE", "DUK", "SO",
    # Real Estate
    "PLD", "AMT",
    # Additional large caps
    "DELL", "BX", "ETN",
]

assert len(TICKERS) == 100, f"Expected 100 tickers, got {len(TICKERS)}"

# ETFs utilisés comme benchmarks et pour l'analyse d'exposition sectorielle
# SPY = marché global, les 11 SPDR sectoriels couvrent les secteurs GICS
BENCHMARK_TICKERS = [
    "SPY",   # S&P 500 (marché global)
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLU",   # Utilities
    "XLC",   # Communication Services 
    ]