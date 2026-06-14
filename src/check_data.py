"""
check_data.py

Vérifie la qualité des données téléchargées dans data/prices.parquet
et data/benchmarks.parquet :
- shape, dates couvertes
- tickers manquants par rapport à l'univers attendu
- valeurs manquantes / NaN
- doublons
- trous dans les séries temporelles (jours ouvrés manquants)

Usage:
    python src/check_data.py
"""

from pathlib import Path

import pandas as pd

from tickers import TICKERS, BENCHMARK_TICKERS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_FILE = DATA_DIR / "prices.parquet"
BENCHMARKS_FILE = DATA_DIR / "benchmarks.parquet"


def check_dataset(df: pd.DataFrame, expected_tickers: list[str], name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Checking {name}")
    print(f"{'=' * 60}")

    # 1. Shape & dates
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Columns: {df.columns.tolist()}")

    # 2. Tickers manquants
    downloaded = set(df["ticker"].unique())
    expected = set(expected_tickers)
    missing = expected - downloaded
    extra = downloaded - expected

    print(f"\nExpected tickers: {len(expected)}")
    print(f"Downloaded tickers: {len(downloaded)}")
    if missing:
        print(f"MISSING tickers ({len(missing)}): {sorted(missing)}")
    else:
        print("No missing tickers.")
    if extra:
        print(f"UNEXPECTED tickers ({len(extra)}): {sorted(extra)}")

    # 3. Valeurs manquantes / NaN
    print("\nNaN counts per column:")
    nan_counts = df.isna().sum()
    print(nan_counts[nan_counts > 0] if nan_counts.sum() > 0 else "No NaN values.")

    # Lignes avec prix négatifs ou nuls (suspect)
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]
    invalid_prices = df[(df[price_cols] <= 0).any(axis=1)]
    if not invalid_prices.empty:
        print(f"\nWARNING: {len(invalid_prices)} rows with price <= 0")
        print(invalid_prices[["date", "ticker"] + price_cols].head())

    # 4. Doublons
    dupes = df.duplicated(subset=["date", "ticker"]).sum()
    print(f"\nDuplicate (date, ticker) rows: {dupes}")

    # 5. Trous dans les séries temporelles par ticker
    print("\nChecking for gaps in time series (per ticker)...")
    gap_report = []
    for ticker, group in df.groupby("ticker"):
        dates = pd.to_datetime(group["date"]).sort_values()
        # jours ouvrés attendus entre min et max
        expected_days = pd.bdate_range(dates.min(), dates.max())
        missing_days = expected_days.difference(dates)
        n_rows = len(dates)
        n_expected = len(expected_days)
        coverage = n_rows / n_expected if n_expected > 0 else 0
        if coverage < 0.95:  # tolérance pour jours fériés
            gap_report.append((ticker, n_rows, n_expected, coverage, len(missing_days)))

    if gap_report:
        print(f"\nTickers with significant gaps (<95% coverage of business days):")
        gap_df = pd.DataFrame(
            gap_report,
            columns=["ticker", "n_rows", "n_expected_bdays", "coverage", "n_missing_days"],
        )
        print(gap_df.sort_values("coverage").to_string(index=False))
    else:
        print("No significant gaps detected (>=95% business day coverage for all tickers).")


def main():
    if not PRICES_FILE.exists():
        raise FileNotFoundError(
            f"{PRICES_FILE} not found. Run `python src/data_loader.py` first."
        )

    prices = pd.read_parquet(PRICES_FILE)
    check_dataset(prices, TICKERS, "prices.parquet (100 stocks)")

    if BENCHMARKS_FILE.exists():
        benchmarks = pd.read_parquet(BENCHMARKS_FILE)
        check_dataset(benchmarks, BENCHMARK_TICKERS, "benchmarks.parquet (ETFs)")
    else:
        print(f"\n{BENCHMARKS_FILE} not found, skipping.")


if __name__ == "__main__":
    main()
