"""
visualize_metrics.py

Visualise le tableau de métriques des portefeuilles benchmarks
(data/portfolio_metrics.parquet) sous forme de graphiques comparatifs.

Usage:
    python src/visualize_metrics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METRICS_FILE = DATA_DIR / "portfolio_metrics.parquet"
RETURNS_FILE = DATA_DIR / "portfolio_returns.parquet"
PLOTS_DIR = DATA_DIR / "plots"

PORTFOLIO_COLORS = {
    "equal_weight":   "#378ADD",
    "momentum_top20": "#0F6E56",
    "low_vol_top20":  "#BA7517",
    "spy":            "#5F5E5A",
}

PORTFOLIO_LABELS = {
    "equal_weight":   "Equal Weight",
    "momentum_top20": "Momentum Top 20%",
    "low_vol_top20":  "Low Vol Top 20%",
    "spy":            "SPY",
}


def plot_metrics_bar(metrics: pd.DataFrame) -> None:
    """
    Graphique en barres comparatif pour chaque métrique clé.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Benchmark Portfolio Performance Comparison", fontsize=15)

    metric_configs = [
        ("annual_return",     "Annual Return",      True,  "{:.1%}"),
        ("annual_volatility", "Annual Volatility",  False, "{:.1%}"),
        ("sharpe_ratio",      "Sharpe Ratio",       True,  "{:.2f}"),
        ("max_drawdown",      "Max Drawdown",       False, "{:.1%}"),
        ("calmar_ratio",      "Calmar Ratio",       True,  "{:.2f}"),
        ("cumulative_return", "Cumulative Return",  True,  "{:.1%}"),
    ]

    portfolios = metrics.index.tolist()
    colors = [PORTFOLIO_COLORS.get(p, "#999") for p in portfolios]
    labels = [PORTFOLIO_LABELS.get(p, p) for p in portfolios]

    for ax, (col, title, higher_better, fmt) in zip(axes.flat, metric_configs):
        values = metrics[col].values.astype(float)

        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

        # Annotation des valeurs sur chaque barre
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            y_pos = bar.get_height() + (abs(bar.get_height()) * 0.02)
            if val < 0:
                y_pos = bar.get_height() - (abs(bar.get_height()) * 0.08)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                fmt.format(val),
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Higher is better ✓" if higher_better else "Lower is better ✓",
                       fontsize=8, color="gray")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: f"{x:.0%}" if "%" in fmt else f"{x:.2f}"
            )
        )
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "metrics_comparison.png"
    plt.savefig(out, dpi=120)
    print(f"Saved to {out}")
    plt.close(fig)


def plot_cumulative_returns(returns: pd.DataFrame) -> None:
    """Rendements cumulés + drawdown côte à côte."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Cumulative Returns & Drawdown (2015–2024)", fontsize=14)

    ax1 = axes[0]
    ax2 = axes[1]

    for col in returns.columns:
        clean = returns[col].dropna()
        color = PORTFOLIO_COLORS.get(col, "#999")
        label = PORTFOLIO_LABELS.get(col, col)

        # Rendements cumulés
        cumret = (1 + clean).cumprod()
        ax1.plot(cumret.index, cumret.values, label=label,
                 color=color, linewidth=1.8)

        # Drawdown
        rolling_max = cumret.cummax()
        drawdown = (cumret - rolling_max) / rolling_max
        ax2.plot(drawdown.index, drawdown.values * 100,
                 label=label, color=color, linewidth=1.2)

    ax1.set_ylabel("Portfolio Value (base 1)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_ylabel("Drawdown (%)")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                     alpha=0.05, color="red")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "cumulative_returns.png"
    plt.savefig(out, dpi=120)
    print(f"Saved to {out}")
    plt.close(fig)


def print_metrics_table(metrics: pd.DataFrame) -> None:
    """Affiche le tableau de métriques formaté dans le terminal."""
    display = metrics.copy()

    pct_cols = ["cumulative_return", "annual_return", "annual_volatility", "max_drawdown"]
    ratio_cols = ["sharpe_ratio", "calmar_ratio"]

    for col in pct_cols:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
    for col in ratio_cols:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )

    display.index = [PORTFOLIO_LABELS.get(i, i) for i in display.index]

    print("\n" + "=" * 70)
    print("BENCHMARK PORTFOLIO METRICS")
    print("=" * 70)
    print(display.drop(columns=["n_trading_days"], errors="ignore").to_string())
    print("=" * 70)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {METRICS_FILE}...")
    metrics = pd.read_parquet(METRICS_FILE)

    print(f"Loading {RETURNS_FILE}...")
    returns = pd.read_parquet(RETURNS_FILE)
    returns.index = pd.to_datetime(returns.index)

    print_metrics_table(metrics)

    print("\nGenerating charts...")
    plot_metrics_bar(metrics)
    plot_cumulative_returns(returns)

    print(f"\nAll plots saved in {PLOTS_DIR}/")


if __name__ == "__main__":
    main()