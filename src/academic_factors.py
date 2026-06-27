"""
academic_factors.py

Construction des scores de facteurs académiques inspirés de Fama-French
et de la littérature quantitative moderne :

- Value      : actions "bon marché" vs "chères" (P/E, P/B faibles)
- Quality    : entreprises rentables et solides (ROE, marges, dette faible)
- Size       : petites capitalisations vs grandes
- Low Vol    : actions peu volatiles (dérivé de prices.parquet, point-in-time parfait)
- Momentum   : actions en tendance haussière (dérivé de features.parquet)

Méthodologie : ranking cross-sectionnel à chaque date (ou snapshot pour les
fondamentaux), normalisé en z-score entre -1 et +1 via percentile rank.

Les scores sont sauvegardés dans data/academic_factors.parquet.
Ils seront utilisés en semaine 5 pour construire des portefeuilles factoriels
long/short (top 20% vs bottom 20%) comme benchmarks de comparaison.

NOTES SUR LES LIMITES :
- Value et Quality utilisent un snapshot fondamental statique (yfinance .info).
  En production, on utiliserait des données historiques point-in-time.
- Low Vol et Momentum sont dérivés des prix → pas de look-ahead bias.

Usage:
    python src/academic_factors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_FILE = DATA_DIR / "features.parquet"
FUNDAMENTALS_FILE = DATA_DIR / "fundamentals.parquet"
OUTPUT_FILE = DATA_DIR / "academic_factors.parquet"


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------

def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """
    Transforme une série de valeurs en percentile rank cross-sectionnel
    entre 0 et 1, puis en z-score centré en 0 (entre ~-1.5 et +1.5).
    Gère les NaN en les ignorant du ranking.

    Ex: sur 100 actions, l'action au 90ème percentile aura un score de +0.8.
    """
    return series.rank(pct=True, na_option="keep") * 2 - 1


def zscore(series: pd.Series) -> pd.Series:
    """Z-score standard : (x - mean) / std. Alternatif au percentile rank."""
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return (series - mean) / std


# ----------------------------------------------------------------------
# Facteurs dérivés des fondamentaux (snapshot statique)
# ----------------------------------------------------------------------

def build_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """
    Value : actions à faible valorisation.
    - P/E faible  → value élevée  (on inverse)
    - P/B faible  → value élevée  (on inverse)
    Score final = moyenne des deux rangs inversés.

    Intuition : une action avec P/E=10 est "bon marché" vs P/E=40.
    """
    scores = pd.DataFrame(index=fund_df["ticker"])

    if "trailing_pe" in fund_df.columns:
        pe = fund_df.set_index("ticker")["trailing_pe"]
        # Filtrer les P/E négatifs (pertes) — non informatifs pour le Value
        pe = pe.where(pe > 0)
        scores["rank_pe"] = cross_sectional_rank(-pe)  # inverser: PE bas = score élevé

    if "price_to_book" in fund_df.columns:
        pb = fund_df.set_index("ticker")["price_to_book"]
        pb = pb.where(pb > 0)
        scores["rank_pb"] = cross_sectional_rank(-pb)

    score = scores.mean(axis=1)
    score.name = "factor_value"
    return score


def build_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """
    Quality : entreprises rentables, marges élevées, peu endettées.
    - ROE élevé      → quality élevée
    - Profit margin élevée → quality élevée
    - Debt/equity faible   → quality élevée (inverser)
    Score final = moyenne des trois rangs.

    Intuition : une entreprise avec ROE=25%, marges=20%, dette faible
    est structurellement plus solide.
    """
    scores = pd.DataFrame(index=fund_df["ticker"])

    if "roe" in fund_df.columns:
        scores["rank_roe"] = cross_sectional_rank(
            fund_df.set_index("ticker")["roe"]
        )

    if "profit_margin" in fund_df.columns:
        scores["rank_margin"] = cross_sectional_rank(
            fund_df.set_index("ticker")["profit_margin"]
        )

    if "debt_to_equity" in fund_df.columns:
        de = fund_df.set_index("ticker")["debt_to_equity"]
        de = de.where(de >= 0)
        scores["rank_debt"] = cross_sectional_rank(-de)  # inverser: dette faible = score élevé

    score = scores.mean(axis=1)
    score.name = "factor_quality"
    return score


def build_size_score(fund_df: pd.DataFrame) -> pd.Series:
    """
    Size : petites capitalisations (small caps) vs grandes (large caps).
    Dans la littérature Fama-French, les small caps surperforment à long terme.
    Score élevé = petite cap (market cap faible).

    Intuition : SMB (Small Minus Big) = rendement small caps - rendement large caps.
    """
    if "market_cap" not in fund_df.columns:
        return pd.Series(dtype=float, name="factor_size")

    mc = fund_df.set_index("ticker")["market_cap"]
    mc = mc.where(mc > 0)
    score = cross_sectional_rank(-mc)  # inverser : petite cap = score élevé
    score.name = "factor_size"
    return score


# ----------------------------------------------------------------------
# Facteurs dérivés des prix (point-in-time parfait, pas de look-ahead bias)
# ----------------------------------------------------------------------

def build_low_vol_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Low Volatility : actions peu volatiles.
    Calculé à chaque date comme le ranking cross-sectionnel de la volatilité
    inverse sur toutes les actions → series temporelle complète, pas un snapshot.

    Intuition : l'anomalie Low Vol montre que les actions peu risquées
    offrent de meilleurs rendements ajustés du risque (Ang et al., 2006).
    """
    if "volatility_20d" not in features_df.columns:
        return pd.DataFrame()

    result = (
        features_df
        .dropna(subset=["volatility_20d"])
        .groupby("date", group_keys=False)
        .apply(lambda g: g.assign(
            factor_low_vol=cross_sectional_rank(-g["volatility_20d"])
        ))
        [["date", "ticker", "factor_low_vol"]]
    )
    return result


def build_momentum_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum factoriel : ranking cross-sectionnel du momentum 60j à chaque date.
    Correspond au facteur UMD (Up Minus Down) de Carhart (1997).

    On utilise le momentum 60j (3 mois) plutôt que 20j car le facteur momentum
    académique est typiquement calculé sur 12 mois avec skip d'1 mois.
    Le 60j est un proxy raisonnable sur données journalières.

    Intuition : les gagnants récents continuent de surperformer (Jegadeesh & Titman, 1993).
    """
    if "momentum_60d" not in features_df.columns:
        return pd.DataFrame()

    result = (
        features_df
        .dropna(subset=["momentum_60d"])
        .groupby("date", group_keys=False)
        .apply(lambda g: g.assign(
            factor_momentum=cross_sectional_rank(g["momentum_60d"])
        ))
        [["date", "ticker", "factor_momentum"]]
    )
    return result


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def build_academic_factors(
    features_df: pd.DataFrame,
    fund_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit tous les facteurs et les fusionne en un seul DataFrame :
    [date, ticker, factor_value, factor_quality, factor_size,
     factor_low_vol, factor_momentum]
    """
    print("Computing Low Volatility factor (time series)...")
    low_vol_df = build_low_vol_scores(features_df)

    print("Computing Momentum factor (time series)...")
    mom_df = build_momentum_scores(features_df)

    # Fusionner les deux séries temporelles
    combined = pd.merge(low_vol_df, mom_df, on=["date", "ticker"], how="outer")

    if not fund_df.empty:
        print("Computing Value factor (snapshot)...")
        value_score = build_value_score(fund_df)

        print("Computing Quality factor (snapshot)...")
        quality_score = build_quality_score(fund_df)

        print("Computing Size factor (snapshot)...")
        size_score = build_size_score(fund_df)

        # Fusionner les scores snapshot (statiques, répétés pour chaque date)
        snapshot_scores = pd.DataFrame({
            "ticker": value_score.index,
            "factor_value": value_score.values,
            "factor_quality": quality_score.reindex(value_score.index).values,
            "factor_size": size_score.reindex(value_score.index).values,
        })

        combined = pd.merge(combined, snapshot_scores, on="ticker", how="left")
    else:
        print("No fundamental data found. Skipping Value, Quality, Size factors.")
        combined["factor_value"] = np.nan
        combined["factor_quality"] = np.nan
        combined["factor_size"] = np.nan

    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    return combined


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Tickers: {df['ticker'].nunique()}")

    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    print(f"\nFactors computed: {factor_cols}")

    print("\nNaN counts per factor:")
    print(df[factor_cols].isna().sum())

    print("\nFactor score statistics (cross-sectional, all dates):")
    print(df[factor_cols].describe().T[["mean", "std", "min", "max"]])


def main():
    print(f"Loading {FEATURES_FILE}...")
    features_df = pd.read_parquet(FEATURES_FILE)
    features_df["date"] = pd.to_datetime(features_df["date"])

    if FUNDAMENTALS_FILE.exists():
        print(f"Loading {FUNDAMENTALS_FILE}...")
        fund_df = pd.read_parquet(FUNDAMENTALS_FILE)
    else:
        print(f"WARNING: {FUNDAMENTALS_FILE} not found.")
        print("Run `python src/fundamentals_loader.py` first for Value/Quality/Size.")
        print("Proceeding with Low Vol and Momentum only.\n")
        fund_df = pd.DataFrame()

    df = build_academic_factors(features_df, fund_df)

    print_summary(df)

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()