"""
Earnings Surprise Analyzer — Main Orchestrator
================================================
Fetches earnings data, computes surprises & returns, runs statistical tests,
generates visualizations, and backtests a simple long/short strategy.

Usage:
    python main.py                  # Run with caching
    python main.py --no-cache       # Force fresh data fetch
    python main.py --no-backtest    # Skip backtest
"""

import argparse
import os
import sys
import warnings

import pandas as pd

# Suppress noisy warnings from yfinance / urllib3
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

import config
from src.data_fetcher import load_or_fetch_earnings, load_or_fetch_prices
from src.surprise_calculator import build_analysis_dataset
from src.analyzer import run_full_analysis
from src.visualizer import generate_all_charts
from src.backtester import run_backtest


def print_banner():
    """Print a styled banner."""
    print()
    print("╔" + "═"*58 + "╗")
    print("║" + " EARNINGS SURPRISE ANALYZER ".center(58) + "║")
    print("║" + " Post-Earnings Announcement Drift (PEAD) Study ".center(58) + "║")
    print("╚" + "═"*58 + "╝")
    print()


def print_data_summary(df: pd.DataFrame):
    """Print a summary of the analysis dataset."""
    print("\n" + "="*60)
    print("  DATA SUMMARY")
    print("="*60)
    
    print(f"\n  Total earnings events:  {len(df)}")
    print(f"  Unique tickers:         {df['ticker'].nunique()}")
    print(f"  Date range:             {df['earnings_date'].min().strftime('%Y-%m-%d')} → "
          f"{df['earnings_date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n  Category breakdown:")
    for cat in ["Beat", "In-Line", "Miss"]:
        count = len(df[df["category"] == cat])
        pct = count / len(df) * 100
        print(f"    {cat:8s}: {count:3d} ({pct:.1f}%)")
    
    print(f"\n  Surprise % statistics:")
    print(f"    Mean:   {df['surprise_pct'].mean():+.2f}%")
    print(f"    Median: {df['surprise_pct'].median():+.2f}%")
    print(f"    Std:    {df['surprise_pct'].std():.2f}%")
    print(f"    Min:    {df['surprise_pct'].min():+.2f}%")
    print(f"    Max:    {df['surprise_pct'].max():+.2f}%")
    
    print(f"\n  Top surprise events:")
    top = df.nlargest(5, "surprise_pct")[["ticker", "earnings_date", "surprise_pct", "ret_day0"]]
    for _, row in top.iterrows():
        print(f"    {row['ticker']:5s} {row['earnings_date'].strftime('%Y-%m-%d')}  "
              f"surprise={row['surprise_pct']:+.1f}%  day0_ret={row['ret_day0']:+.2f}%")
    
    print(f"\n  Worst surprise events:")
    bottom = df.nsmallest(5, "surprise_pct")[["ticker", "earnings_date", "surprise_pct", "ret_day0"]]
    for _, row in bottom.iterrows():
        print(f"    {row['ticker']:5s} {row['earnings_date'].strftime('%Y-%m-%d')}  "
              f"surprise={row['surprise_pct']:+.1f}%  day0_ret={row['ret_day0']:+.2f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Earnings Surprise Analyzer")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force fresh data fetch (ignore cache)")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Skip the backtest")
    args = parser.parse_args()
    
    use_cache = not args.no_cache
    
    print_banner()
    config.ensure_dirs()
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Fetch Data
    # ═══════════════════════════════════════════════════════════════════
    print("STEP 1: Fetching Data")
    print("─"*40)
    
    print("\n[Earnings Data]")
    earnings_df = load_or_fetch_earnings(
        config.TICKERS, use_cache=use_cache, verbose=True
    )
    
    print(f"\n[Price Data]")
    prices_df = load_or_fetch_prices(
        config.TICKERS, use_cache=use_cache, verbose=True
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Build Analysis Dataset
    # ═══════════════════════════════════════════════════════════════════
    print("\n\nSTEP 2: Computing Surprises & Returns")
    print("─"*40)
    
    analysis_df = build_analysis_dataset(earnings_df, prices_df, verbose=True)
    
    # Save merged dataset
    analysis_df.to_csv(config.MERGED_CACHE, index=False)
    print(f"  Saved analysis dataset → {os.path.basename(config.MERGED_CACHE)}")
    
    # Print summary
    print_data_summary(analysis_df)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Statistical Analysis
    # ═══════════════════════════════════════════════════════════════════
    print("\n\nSTEP 3: Running Statistical Analysis")
    print("─"*40)
    
    results = run_full_analysis(analysis_df, verbose=True)
    
    # Save statistical results
    if "t_tests" in results:
        results["t_tests"].to_csv(config.STATS_OUTPUT, index=False)
        print(f"\n  Saved stats → {os.path.basename(config.STATS_OUTPUT)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Backtest
    # ═══════════════════════════════════════════════════════════════════
    backtest_df = None
    backtest_metrics = {}
    
    if not args.no_backtest:
        print("\n\nSTEP 4: Running Backtest")
        print("─"*40)
        
        backtest_df, backtest_metrics = run_backtest(
            analysis_df, prices_df, verbose=True
        )
        
        if not backtest_df.empty:
            backtest_df.to_csv(config.BACKTEST_OUTPUT, index=False)
            print(f"\n  Saved backtest log → {os.path.basename(config.BACKTEST_OUTPUT)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: Visualizations
    # ═══════════════════════════════════════════════════════════════════
    print("\n\nSTEP 5: Generating Visualizations")
    print("─"*40)
    
    chart_paths = generate_all_charts(
        analysis_df, prices_df, backtest_df=backtest_df, verbose=True
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════════════════
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " ANALYSIS COMPLETE ".center(58) + "║")
    print("╚" + "═"*58 + "╝")
    print(f"\n  📊 Charts:  {config.CHARTS_DIR}")
    print(f"  📁 Data:    {config.DATA_DIR}")
    print(f"  📈 Events:  {len(analysis_df)} earnings announcements analyzed")
    
    if backtest_metrics:
        print(f"\n  💰 Strategy: {backtest_metrics.get('total_return_pct', 0):+.2f}% total return | "
              f"Sharpe: {backtest_metrics.get('sharpe_ratio', 0):.2f} | "
              f"Win Rate: {backtest_metrics.get('win_rate_pct', 0):.1f}%")
    
    print()
    
    return analysis_df, results, backtest_df, backtest_metrics


if __name__ == "__main__":
    main()
