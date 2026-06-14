"""
data_loader.py

Télécharge les données OHLCV (Open, High, Low, Close, Adj Close, Volume)
pour l'univers de 100 actions défini dans tickers.py, via yfinance,
et sauvegarde une base de données unique au format parquet.

Usage:
    python src/data_loader.py
"""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from tickers import TICKERS, BENCHMARK_TICKERS

START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_FILE = DATA_DIR / "prices.parquet"
BENCHMARK_OUTPUT_FILE = DATA_DIR / "benchmarks.parquet"


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Télécharge l'historique OHLCV pour chaque ticker et retourne
    un DataFrame long format avec colonnes:
    [date, ticker, open, high, low, close, adj_close, volume]
    """
    all_frames = []
    failed = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Downloading {ticker}...")
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                print(f"  -> No data for {ticker}, skipping.")
                failed.append(ticker)
                continue

            # yfinance peut renvoyer des colonnes multi-index (Ticker, Field)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "adj_close", "volume"]]
            df.index.name = "date"
            df = df.reset_index()
            df["ticker"] = ticker

            all_frames.append(df)

        except Exception as e:
            print(f"  -> Error downloading {ticker}: {e}")
            failed.append(ticker)

        time.sleep(0.3)  # éviter le rate limiting

    if failed:
        print(f"\nFailed tickers ({len(failed)}): {failed}")

    if not all_frames:
        raise RuntimeError("No data downloaded for any ticker.")

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df = full_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return full_df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data for {len(TICKERS)} tickers "
          f"from {START_DATE} to {END_DATE}...\n")

    df = download_prices(TICKERS, START_DATE, END_DATE)

    print(f"\nTotal rows: {len(df)}")
    print(f"Tickers downloaded: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")

    # Téléchargement des ETFs benchmarks (SPY + secteurs SPDR)
    print(f"\nDownloading data for {len(BENCHMARK_TICKERS)} benchmark ETFs "
          f"from {START_DATE} to {END_DATE}...\n")

    bench_df = download_prices(BENCHMARK_TICKERS, START_DATE, END_DATE)

    print(f"\nTotal rows: {len(bench_df)}")
    print(f"ETFs downloaded: {bench_df['ticker'].nunique()}")
    print(f"Date range: {bench_df['date'].min()} to {bench_df['date'].max()}")

    bench_df.to_parquet(BENCHMARK_OUTPUT_FILE, index=False)
    print(f"\nSaved to {BENCHMARK_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
