"""
factors.py

Feature engineering sur les données OHLCV (data/prices.parquet) :
- Momentum (5j, 20j, 60j)
- Volatilité (20j)
- Moyennes mobiles (MA20, MA50, MA200)
- RSI (14j)
- MACD (12, 26, 9)
- Volume relatif (vs moyenne 20j)
- Target : rendement futur à 20 jours (forward return)

Toutes les features sont calculées par ticker, indépendamment, en respectant
la contrainte de non-anticipation (look-ahead bias) : une feature à la date t
n'utilise que des informations disponibles jusqu'à t inclus.

Usage:
    python src/factors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "prices.parquet"
OUTPUT_FILE = DATA_DIR / "features.parquet"

# Paramètres
MOMENTUM_WINDOWS = [5, 20, 60]
VOLATILITY_WINDOW = 20
MA_WINDOWS = [20, 50, 200]
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_WINDOW = 20
TARGET_HORIZON = 20  # rendement futur à 20 jours


# ----------------------------------------------------------------------
# Indicateurs individuels (appliqués à un seul ticker, trié par date)
# ----------------------------------------------------------------------

def add_momentum(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Momentum: P_t / P_{t-k} - 1, pour chaque k dans windows."""
    for k in windows:
        df[f"momentum_{k}d"] = df["close"].pct_change(k)
    return df


def add_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Volatilité: écart-type des rendements quotidiens sur une fenêtre glissante."""
    daily_returns = df["close"].pct_change()
    df[f"volatility_{window}d"] = daily_returns.rolling(window).std()
    return df


def add_moving_averages(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Moyennes mobiles simples du prix de clôture."""
    for w in windows:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        # Distance relative du prix à la moyenne mobile (souvent plus utile que la MA brute)
        df[f"close_to_ma_{w}"] = df["close"] / df[f"ma_{w}"] - 1
    return df


def add_rsi(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Relative Strength Index (RSI), méthode de Wilder simplifiée
    via moyenne mobile simple des gains/pertes.
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Quand avg_loss == 0 et avg_gain > 0 -> RSI = 100
    rsi = rsi.where(avg_loss != 0, 100)

    df[f"rsi_{window}"] = rsi
    return df


def add_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram = MACD - Signal line
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line
    return df


def add_relative_volume(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Volume relatif: volume du jour / moyenne mobile du volume sur `window` jours."""
    avg_volume = df["volume"].rolling(window).mean()
    df[f"volume_relative_{window}d"] = df["volume"] / avg_volume
    return df


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Target: rendement futur à `horizon` jours.
    target_t = close_{t+horizon} / close_t - 1

    ATTENTION: cette colonne utilise des informations FUTURES par construction.
    Elle ne doit JAMAIS être utilisée comme feature, uniquement comme variable
    à prédire (y). Les dernières `horizon` lignes de chaque ticker auront un
    target NaN (pas de futur disponible).
    """
    df[f"target_{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1
    return df


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def compute_features_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    """Applique tous les indicateurs à un DataFrame d'un seul ticker, trié par date."""
    group = group.sort_values("date").reset_index(drop=True)

    group = add_momentum(group, MOMENTUM_WINDOWS)
    group = add_volatility(group, VOLATILITY_WINDOW)
    group = add_moving_averages(group, MA_WINDOWS)
    group = add_rsi(group, RSI_WINDOW)
    group = add_macd(group, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    group = add_relative_volume(group, VOLUME_WINDOW)
    group = add_target(group, TARGET_HORIZON)

    return group


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features pour tous les tickers."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    results = []
    tickers = df["ticker"].unique()
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Computing features for {ticker}...")
        group = df[df["ticker"] == ticker]
        group = compute_features_for_ticker(group)
        results.append(group)

    full_df = pd.concat(results, ignore_index=True)
    full_df = full_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return full_df


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    print(f"Input shape: {df.shape}")
    print(f"Tickers: {df['ticker'].nunique()}")

    features_df = build_features(df)

    print(f"\nOutput shape: {features_df.shape}")
    print(f"New columns: {[c for c in features_df.columns if c not in df.columns]}")

    # Statistiques rapides sur les NaN (attendus en début de série, à cause des
    # fenêtres glissantes, et en fin de série pour le target)
    feature_cols = [c for c in features_df.columns if c not in df.columns]
    print("\nNaN counts per feature (expected: warm-up period + target tail):")
    print(features_df[feature_cols].isna().sum())

    features_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
