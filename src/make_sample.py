"""
make_sample.py

Génère un petit échantillon de data/prices.parquet (ex: 3 tickers, 1 an)
au format CSV, destiné à être versionné sur GitHub pour montrer la structure
des données sans alourdir le dépôt avec le fichier parquet complet.

Usage:
    python src/make_sample.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_FILE = DATA_DIR / "prices.parquet"
SAMPLE_DIR = DATA_DIR / "sample"
SAMPLE_FILE = SAMPLE_DIR / "prices_sample.csv"

SAMPLE_TICKERS = ["AAPL", "MSFT", "JPM"]
SAMPLE_YEAR = 2024


def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PRICES_FILE)
    df["date"] = pd.to_datetime(df["date"])

    sample = df[
        (df["ticker"].isin(SAMPLE_TICKERS))
        & (df["date"].dt.year == SAMPLE_YEAR)
    ]

    sample.to_csv(SAMPLE_FILE, index=False)
    print(f"Sample saved to {SAMPLE_FILE} ({len(sample)} rows, "
          f"tickers={SAMPLE_TICKERS}, year={SAMPLE_YEAR})")


if __name__ == "__main__":
    main()
