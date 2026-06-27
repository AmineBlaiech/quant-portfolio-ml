"""
benchmarks.py

Construction et comparaison de portefeuilles benchmarks (semaine 5) :

1. Equal Weight      : 100 actions équipondérées, rebalancé mensuellement
2. Momentum          : top 20% sur factor_momentum, rebalancé mensuellement
3. Low Volatility    : top 20% sur factor_low_vol, rebalancé mensuellement
4. SPY               : ETF S&P 500 (benchmark marché passif)

Métriques calculées pour chaque portefeuille :
- Rendement cumulé
- Rendement annualisé
- Volatilité annualisée
- Sharpe Ratio (taux sans risque = 0% par simplification)
- Maximum Drawdown
- Calmar Ratio (rendement annualisé / max drawdown)

Résultats sauvegardés dans :
- data/portfolio_returns.parquet  (rendements journaliers de chaque portefeuille)
- data/portfolio_metrics.parquet  (métriques de performance)

Usage:
    python src/benchmarks.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_FILE = DATA_DIR / "prices.parquet"
BENCHMARKS_FILE = DATA_DIR / "benchmarks.parquet"
ACADEMIC_FILE = DATA_DIR / "academic_factors.parquet"
RETURNS_OUTPUT = DATA_DIR / "portfolio_returns.parquet"
METRICS_OUTPUT = DATA_DIR / "portfolio_metrics.parquet"
PLOTS_DIR = DATA_DIR / "plots"

REBALANCE_FREQ = "ME"       # fin de mois (Month End)
TOP_QUANTILE = 0.20          # top 20% pour les portefeuilles factoriels
RISK_FREE_RATE = 0.0         # simplification : taux sans risque = 0%
TRADING_DAYS_PER_YEAR = 252


# ----------------------------------------------------------------------
# Calcul des rendements journaliers
# ----------------------------------------------------------------------

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements journaliers de chaque ticker à partir de adj_close.
    Retourne un DataFrame wide : index=date, colonnes=tickers.
    """
    pivot = prices.pivot(index="date", columns="ticker", values="adj_close")
    pivot = pivot.sort_index()
    returns = pivot.pct_change()
    return returns


# ----------------------------------------------------------------------
# Construction des portefeuilles
# ----------------------------------------------------------------------

def build_equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Portefeuille équipondéré : poids identiques (1/N) pour toutes les actions.
    Rebalancé mensuellement.
    Référence : benchmark naïf mais difficile à battre (DeMiguel et al., 2009).
    """
    # Rebalancer mensuellement = recalculer les poids à chaque fin de mois
    # En équipondéré, les poids ne changent pas, mais on filtre les
    # tickers disponibles à chaque date
    portfolio_returns = returns.mean(axis=1)  # moyenne cross-sectionnelle = 1/N
    portfolio_returns.name = "equal_weight"
    return portfolio_returns


def build_factor_portfolio(
    returns: pd.DataFrame,
    factor_scores: pd.DataFrame,
    factor_col: str,
    name: str,
    top_quantile: float = TOP_QUANTILE,
) -> pd.Series:
    """
    Portefeuille factoriel long-only : à chaque rebalancement mensuel,
    sélectionne le top `top_quantile` des actions selon le score factoriel,
    les équipondère, et détient jusqu'au prochain rebalancement.

    Long-only (pas de short) pour rester simple et réaliste pour un
    investisseur retail. En hedge fund, on ferait long/short.
    """
    # Dates de rebalancement (fins de mois présentes dans les données)
    rebal_dates = pd.date_range(
        start=returns.index.min(),
        end=returns.index.max(),
        freq=REBALANCE_FREQ,
    )

    portfolio_returns = pd.Series(index=returns.index, dtype=float, name=name)

    all_trading_dates = returns.index.tolist()

    for i, rebal_date in enumerate(rebal_dates):
        # Date effective de rebalancement (plus proche date de trading ≤ rebal_date)
        trading_dates_before = [d for d in all_trading_dates if d <= rebal_date]
        if not trading_dates_before:
            continue
        effective_rebal = trading_dates_before[-1]

        # Date de fin de la période (prochain rebalancement ou fin des données)
        if i + 1 < len(rebal_dates):
            next_trading = [d for d in all_trading_dates if d > rebal_date]
            if not next_trading:
                continue
            next_rebal = rebal_dates[i + 1]
            period_end_dates = [d for d in all_trading_dates
                                if rebal_date < d <= next_rebal]
        else:
            period_end_dates = [d for d in all_trading_dates if d > rebal_date]

        if not period_end_dates:
            continue

        # Scores factoriels à la date de rebalancement
        scores_at_rebal = factor_scores[
            factor_scores["date"] == effective_rebal
        ][["ticker", factor_col]].dropna()

        if scores_at_rebal.empty:
            continue

        # Sélection du top quantile
        threshold = scores_at_rebal[factor_col].quantile(1 - top_quantile)
        selected = scores_at_rebal[
            scores_at_rebal[factor_col] >= threshold
        ]["ticker"].tolist()

        if not selected:
            continue

        # Rendements équipondérés pour les tickers sélectionnés sur la période
        tickers_available = [t for t in selected if t in returns.columns]
        if not tickers_available:
            continue

        period_returns = returns.loc[
            returns.index.isin(period_end_dates), tickers_available
        ].mean(axis=1)

        portfolio_returns.loc[period_returns.index] = period_returns.values

    return portfolio_returns


def build_spy_portfolio(benchmarks: pd.DataFrame) -> pd.Series:
    """Rendements journaliers de SPY (benchmark marché passif)."""
    spy = benchmarks[benchmarks["ticker"] == "SPY"].copy()
    spy = spy.sort_values("date").set_index("date")
    spy_returns = spy["adj_close"].pct_change()
    spy_returns.name = "spy"
    return spy_returns


# ----------------------------------------------------------------------
# Métriques de performance
# ----------------------------------------------------------------------

def compute_metrics(returns: pd.Series) -> dict:
    """
    Calcule les métriques de performance standard d'un portefeuille.
    """
    clean = returns.dropna()

    if len(clean) == 0:
        return {}

    # Rendement cumulé
    cumulative_return = (1 + clean).prod() - 1

    # Rendement annualisé (CAGR)
    n_years = len(clean) / TRADING_DAYS_PER_YEAR
    annual_return = (1 + cumulative_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    # Volatilité annualisée
    annual_vol = clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe Ratio
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan

    # Maximum Drawdown
    cumulative = (1 + clean).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        "cumulative_return": round(cumulative_return, 4),
        "annual_return": round(annual_return, 4),
        "annual_volatility": round(annual_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4),
        "calmar_ratio": round(calmar, 4),
        "n_trading_days": len(clean),
    }


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------

def plot_cumulative_returns(portfolio_returns: pd.DataFrame) -> None:
    """Graphique des rendements cumulés de tous les portefeuilles."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Rendements cumulés
    ax = axes[0]
    for col in portfolio_returns.columns:
        cumret = (1 + portfolio_returns[col].dropna()).cumprod()
        ax.plot(cumret.index, cumret.values, label=col, linewidth=1.5)
    ax.set_title("Cumulative Returns — Benchmark Portfolios (2015–2024)")
    ax.set_ylabel("Portfolio Value (base 1)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Drawdown
    ax = axes[1]
    for col in portfolio_returns.columns:
        clean = portfolio_returns[col].dropna()
        cumulative = (1 + clean).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        ax.plot(drawdown.index, drawdown.values, label=col, linewidth=1.2)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = PLOTS_DIR / "benchmark_performance.png"
    plt.savefig(output_path, dpi=100)
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def main():
    print("Loading data...")
    prices = pd.read_parquet(PRICES_FILE)
    prices["date"] = pd.to_datetime(prices["date"])

    benchmarks = pd.read_parquet(BENCHMARKS_FILE)
    benchmarks["date"] = pd.to_datetime(benchmarks["date"])

    academic = pd.read_parquet(ACADEMIC_FILE)
    academic["date"] = pd.to_datetime(academic["date"])

    print("Computing daily returns...")
    returns = compute_daily_returns(prices)

    print("\nBuilding portfolios...")

    # 1. Equal Weight
    print("  [1/4] Equal Weight...")
    ew = build_equal_weight_portfolio(returns)

    # 2. Momentum
    print("  [2/4] Momentum (top 20%)...")
    mom = build_factor_portfolio(
        returns, academic, "factor_momentum", "momentum_top20"
    )

    # 3. Low Volatility
    print("  [3/4] Low Volatility (top 20%)...")
    low_vol = build_factor_portfolio(
        returns, academic, "factor_low_vol", "low_vol_top20"
    )

    # 4. SPY
    print("  [4/4] SPY...")
    spy = build_spy_portfolio(benchmarks)

    # Combiner tous les rendements
    portfolio_returns = pd.DataFrame({
        "equal_weight": ew,
        "momentum_top20": mom,
        "low_vol_top20": low_vol,
        "spy": spy,
    }).sort_index()

    # Aligner sur la même période
    start = portfolio_returns.dropna(how="all").index.min()
    end = portfolio_returns.dropna(how="all").index.max()
    portfolio_returns = portfolio_returns.loc[start:end]

    # Métriques
    print("\nComputing performance metrics...")
    metrics = {}
    for col in portfolio_returns.columns:
        metrics[col] = compute_metrics(portfolio_returns[col])

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "portfolio"

    # Affichage
    print("\n" + "=" * 60)
    print("BENCHMARK PORTFOLIO PERFORMANCE")
    print("=" * 60)
    print(metrics_df.to_string())

    # Sauvegarde
    portfolio_returns.to_parquet(RETURNS_OUTPUT)
    print(f"\nReturns saved to {RETURNS_OUTPUT}")

    metrics_df.to_parquet(METRICS_OUTPUT)
    print(f"Metrics saved to {METRICS_OUTPUT}")

    # Graphique
    plot_cumulative_returns(portfolio_returns)


if __name__ == "__main__":
    main()

 