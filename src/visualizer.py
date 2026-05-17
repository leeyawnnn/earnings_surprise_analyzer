"""
Visualization Module
Generates publication-quality charts for earnings surprise analysis.
Uses a consistent dark theme with professional aesthetics.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config


def _setup_style():
    """Set up the global matplotlib style."""
    plt.style.use(config.CHART_STYLE)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.facecolor": config.COLORS["bg_card"],
        "figure.facecolor": config.COLORS["bg_dark"],
        "savefig.facecolor": config.COLORS["bg_dark"],
        "axes.edgecolor": config.COLORS["grid"],
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": config.COLORS["grid"],
        "text.color": config.COLORS["text"],
        "axes.labelcolor": config.COLORS["text"],
        "xtick.color": config.COLORS["text"],
        "ytick.color": config.COLORS["text"],
    })


def _get_category_color(cat: str) -> str:
    """Map category to color."""
    return {
        "Beat": config.COLORS["beat"],
        "Miss": config.COLORS["miss"],
        "In-Line": config.COLORS["inline"],
    }.get(cat, config.COLORS["accent"])


def _save_fig(fig, name: str):
    """Save figure to the charts directory."""
    path = os.path.join(config.CHARTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"    → Saved: {name}.png")
    return path


def plot_scatter_surprise_vs_return(df: pd.DataFrame) -> str:
    """
    Chart 1: Scatter plot of Earnings Surprise % vs Day-0 Return %.
    Color-coded by category with regression line.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=config.CHART_FIGSIZE)
    
    for cat in ["Beat", "In-Line", "Miss"]:
        subset = df[df["category"] == cat]
        ax.scatter(
            subset["surprise_pct"],
            subset["ret_day0"],
            c=_get_category_color(cat),
            label=cat,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5,
            zorder=3,
        )
    
    # Regression line
    valid = df[["surprise_pct", "ret_day0"]].dropna()
    if len(valid) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid["surprise_pct"], valid["ret_day0"]
        )
        x_line = np.linspace(valid["surprise_pct"].min(), valid["surprise_pct"].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=config.COLORS["accent"], linewidth=2,
                linestyle="--", alpha=0.8, zorder=4,
                label=f"OLS fit (R²={r_value**2:.3f}, p={p_value:.4f})")
    
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel("Earnings Surprise (%)")
    ax.set_ylabel("Day-0 Stock Return (%)")
    ax.set_title("Earnings Surprise vs. Announcement-Day Return")
    ax.legend(loc="upper left", framealpha=0.8, facecolor=config.COLORS["bg_card"])
    
    return _save_fig(fig, "scatter_surprise_vs_return")


def plot_bar_avg_returns(df: pd.DataFrame) -> str:
    """
    Chart 2: Bar chart of average returns by surprise category.
    Grouped bars for Day-0, Day-5, Day-20.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=config.CHART_FIGSIZE)
    
    categories = ["Beat", "In-Line", "Miss"]
    windows = ["ret_day0", "ret_day5", "ret_day20"]
    window_labels = ["Day 0", "Day 1-5", "Day 1-20"]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bar_colors = [config.COLORS["accent"], "#7C4DFF", "#FF6E40"]
    
    for i, (col, label) in enumerate(zip(windows, window_labels)):
        means = []
        errors = []
        for cat in categories:
            vals = df[df["category"] == cat][col].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            se = vals.std() / np.sqrt(len(vals)) * 1.96 if len(vals) > 1 else 0
            errors.append(se)
        
        bars = ax.bar(
            x + (i - 1) * width,
            means,
            width,
            label=label,
            color=bar_colors[i],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            yerr=errors,
            capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5, "alpha": 0.7},
            zorder=3,
        )
    
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Earnings Surprise Category")
    ax.set_ylabel("Average Return (%)")
    ax.set_title("Average Post-Earnings Return by Surprise Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.8, facecolor=config.COLORS["bg_card"])
    
    # Add value labels on bars
    for container in ax.containers:
        if hasattr(container, "datavalues"):
            ax.bar_label(container, fmt="%+.2f%%", padding=5, fontsize=9, color=config.COLORS["text"])
    
    return _save_fig(fig, "bar_avg_returns")


def plot_cumulative_returns(df: pd.DataFrame, prices_df: pd.DataFrame) -> str:
    """
    Chart 3: Cumulative returns of 'Beat' vs 'Miss' portfolios over time.
    Builds equal-weight portfolios from earnings events.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=config.CHART_FIGSIZE)
    
    prices_df.index = pd.to_datetime(prices_df.index).normalize()
    
    for cat, color, ls in [("Beat", config.COLORS["beat"], "-"),
                           ("Miss", config.COLORS["miss"], "-"),
                           ("In-Line", config.COLORS["inline"], "--")]:
        events = df[df["category"] == cat].sort_values("earnings_date")
        
        if events.empty:
            continue
        
        # For each event, compute daily returns for 20 days after
        all_daily_returns = []
        
        for _, event in events.iterrows():
            ticker = event["ticker"]
            earn_date = pd.Timestamp(event.get("event_trading_date", event["earnings_date"])).normalize()
            
            if ticker not in prices_df.columns:
                continue
            
            trading_dates = prices_df.index
            mask = trading_dates >= earn_date
            future_dates = trading_dates[mask][:21]  # event day + 20 days
            
            if len(future_dates) < 2:
                continue
            
            event_prices = prices_df.loc[future_dates, ticker].dropna()
            if len(event_prices) < 2:
                continue
            
            daily_rets = event_prices.pct_change().iloc[1:]  # skip first NaN
            daily_rets.index = range(1, len(daily_rets) + 1)  # re-index to day number
            all_daily_returns.append(daily_rets)
        
        if not all_daily_returns:
            continue
        
        # Average daily returns across all events, then cumulate
        combined = pd.DataFrame(all_daily_returns).T
        avg_daily = combined.mean(axis=1)
        cum_ret = (1 + avg_daily).cumprod() - 1
        cum_ret = cum_ret * 100  # to percentage
        
        # Prepend day 0 at 0%
        cum_ret = pd.concat([pd.Series([0.0], index=[0]), cum_ret])
        
        ax.plot(cum_ret.index, cum_ret.values, color=color, linewidth=2.5,
                label=f"{cat} Portfolio (n={len(events)})", linestyle=ls, alpha=0.9)
        ax.fill_between(cum_ret.index, 0, cum_ret.values, color=color, alpha=0.08)
    
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Trading Days After Earnings Announcement")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Post-Earnings Cumulative Returns: Beat vs Miss Portfolios")
    ax.legend(framealpha=0.8, facecolor=config.COLORS["bg_card"])
    ax.set_xlim(0, 20)
    
    return _save_fig(fig, "cumulative_returns")


def plot_return_distributions(df: pd.DataFrame) -> str:
    """
    Chart 4: Overlapping KDE distributions of Day-0 returns for Beat vs Miss.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, col, title in zip(axes, ["ret_day0", "ret_day5"],
                               ["Day-0 Return (%)", "Day 1-5 Return (%)"]):
        for cat in ["Beat", "Miss"]:
            vals = df[df["category"] == cat][col].dropna()
            if len(vals) < 2:
                continue
            
            color = _get_category_color(cat)
            
            ax.hist(vals, bins=20, alpha=0.3, color=color, density=True,
                    edgecolor="white", linewidth=0.5, label=f"{cat} (n={len(vals)})")
            
            # KDE overlay
            try:
                kde_x = np.linspace(vals.min() - 2, vals.max() + 2, 200)
                kde = stats.gaussian_kde(vals)
                ax.plot(kde_x, kde(kde_x), color=color, linewidth=2.5, alpha=0.9)
            except Exception:
                pass
            
            # Mean line
            ax.axvline(vals.mean(), color=color, linewidth=2, linestyle="--", alpha=0.8)
        
        ax.axvline(0, color="white", linewidth=0.5, alpha=0.3)
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {title}")
        ax.legend(framealpha=0.8, facecolor=config.COLORS["bg_card"])
    
    fig.suptitle("Return Distributions: Beat vs Miss", fontsize=16, fontweight="bold",
                 color=config.COLORS["text"], y=1.02)
    fig.tight_layout()
    
    return _save_fig(fig, "return_distributions")


def plot_surprise_heatmap(df: pd.DataFrame) -> str:
    """
    Chart 5: Heatmap of average Day-0 return by surprise quintile and ticker.
    """
    _setup_style()
    
    # Create surprise quintiles
    df_copy = df.copy()
    df_copy["surprise_quintile"] = pd.qcut(
        df_copy["surprise_pct"], q=5, labels=["Q1\n(Worst)", "Q2", "Q3", "Q4", "Q5\n(Best)"],
        duplicates="drop"
    )
    
    # Pivot: quintile vs return metrics
    ret_cols = ["ret_day0", "ret_day5", "ret_day10", "ret_day20"]
    ret_labels = ["Day 0", "Day 1-5", "Day 1-10", "Day 1-20"]
    available_cols = [c for c in ret_cols if c in df_copy.columns]
    available_labels = [l for c, l in zip(ret_cols, ret_labels) if c in df_copy.columns]
    
    pivot_data = []
    for col, label in zip(available_cols, available_labels):
        group = df_copy.groupby("surprise_quintile")[col].mean()
        group.name = label
        pivot_data.append(group)
    
    if not pivot_data:
        return ""
    
    heatmap_df = pd.DataFrame(pivot_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt="+.2f",
        cmap="RdYlGn",
        center=0,
        linewidths=1,
        linecolor=config.COLORS["bg_dark"],
        ax=ax,
        cbar_kws={"label": "Average Return (%)"},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    
    ax.set_title("Average Return by Surprise Quintile and Holding Period",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Surprise Quintile", fontsize=12)
    ax.set_ylabel("Holding Period", fontsize=12)
    
    return _save_fig(fig, "surprise_heatmap")


def plot_backtest_equity(backtest_df: pd.DataFrame) -> str:
    """
    Chart 6: Backtest equity curve for the long-beat/short-miss strategy.
    Uses the portfolio-level equity curve (one point per entry date) to avoid
    vertical stacks from same-day trades.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # ── Equity curve ─────────────────────────────────────────────────────
    ax1 = axes[0]

    equity = backtest_df.attrs.get("equity_curve")
    if equity is None and "cumulative_return" in backtest_df.columns:
        equity = (
            backtest_df.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")[
                ["date", "cumulative_return", "benchmark_cumulative"]
            ]
            .reset_index(drop=True)
        )

    if equity is not None and not equity.empty:
        ax1.plot(equity["date"], equity["cumulative_return"],
                 color=config.COLORS["accent"], linewidth=2.5,
                 marker="o", markersize=5,
                 label="Strategy (equal-weight portfolio)", zorder=3)

        if "benchmark_cumulative" in equity.columns:
            ax1.plot(equity["date"], equity["benchmark_cumulative"],
                     color=config.COLORS["text"], linewidth=1.5, alpha=0.5,
                     linestyle="--", label="SPY Buy & Hold", zorder=2)

        ax1.fill_between(
            equity["date"], 0, equity["cumulative_return"],
            where=equity["cumulative_return"] >= 0,
            color=config.COLORS["beat"], alpha=0.1,
        )
        ax1.fill_between(
            equity["date"], 0, equity["cumulative_return"],
            where=equity["cumulative_return"] < 0,
            color=config.COLORS["miss"], alpha=0.1,
        )

    ax1.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("Backtest: Long Beats / Short Misses Strategy", fontsize=16)
    ax1.legend(framealpha=0.8, facecolor=config.COLORS["bg_card"])
    
    # ── Trade returns ────────────────────────────────────────────────────
    ax2 = axes[1]
    
    if "trade_return" in backtest_df.columns:
        trade_df = backtest_df[backtest_df["trade_return"].notna()]
        colors = [config.COLORS["beat"] if r >= 0 else config.COLORS["miss"]
                  for r in trade_df["trade_return"]]
        ax2.bar(trade_df["date"], trade_df["trade_return"],
                color=colors, alpha=0.7, width=5, edgecolor="none")
    
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Trade Return (%)")
    ax2.set_title("Individual Trade Returns", fontsize=12)
    
    fig.tight_layout()
    
    return _save_fig(fig, "backtest_equity")


def generate_all_charts(
    df: pd.DataFrame,
    prices_df: pd.DataFrame,
    backtest_df: pd.DataFrame = None,
    verbose: bool = True,
) -> list[str]:
    """Generate all visualization charts."""
    config.ensure_dirs()
    
    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)
    
    paths = []
    
    print("\n  1. Scatter: Surprise % vs Return %")
    paths.append(plot_scatter_surprise_vs_return(df))
    
    print("  2. Bar: Average Returns by Category")
    paths.append(plot_bar_avg_returns(df))
    
    print("  3. Cumulative Returns: Beat vs Miss Portfolios")
    paths.append(plot_cumulative_returns(df, prices_df))
    
    print("  4. Return Distributions")
    paths.append(plot_return_distributions(df))
    
    print("  5. Surprise Quintile Heatmap")
    paths.append(plot_surprise_heatmap(df))
    
    if backtest_df is not None and not backtest_df.empty:
        print("  6. Backtest Equity Curve")
        paths.append(plot_backtest_equity(backtest_df))
    
    print(f"\n  ✓ {len(paths)} charts saved to {config.CHARTS_DIR}")
    
    return paths
