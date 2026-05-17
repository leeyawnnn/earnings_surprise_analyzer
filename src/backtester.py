"""
Backtester Module
Implements a simple long-beat / short-miss earnings strategy and evaluates performance.
"""

import numpy as np
import pandas as pd

import config


def run_backtest(
    df: pd.DataFrame,
    prices_df: pd.DataFrame,
    hold_days: int = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest a simple earnings surprise strategy:
    
    - On each earnings announcement:
      - If BEAT  → go LONG at next trading day's open (use close as proxy), hold for N days
      - If MISS  → go SHORT at next trading day's open, hold for N days
      - If IN-LINE → no position
    
    Args:
        df: Analysis dataset with categories and returns
        prices_df: Price data
        hold_days: Number of days to hold each position
        verbose: Print results
    
    Returns:
        (trade_log DataFrame, performance_metrics dict)
    """
    hold_days = hold_days or config.BACKTEST_HOLD_DAYS
    
    prices_df.index = pd.to_datetime(prices_df.index).normalize()
    trading_dates = prices_df.index.sort_values()
    date_to_pos = {d: i for i, d in enumerate(trading_dates)}
    
    trades = []
    
    # Process each earnings event
    events = df[df["category"].isin(["Beat", "Miss"])].sort_values("earnings_date")
    
    for _, event in events.iterrows():
        ticker = event["ticker"]
        category = event["category"]
        earn_date = pd.Timestamp(event.get("event_trading_date", event["earnings_date"])).normalize()
        
        if ticker not in prices_df.columns:
            continue
        
        # Find the next trading day after earnings (entry day)
        future = trading_dates[trading_dates > earn_date]
        if len(future) == 0:
            continue
        
        entry_date = future[0]
        entry_pos = date_to_pos[entry_date]
        
        # Exit after hold_days trading days
        exit_pos = entry_pos + hold_days
        if exit_pos >= len(trading_dates):
            continue
        
        exit_date = trading_dates[exit_pos]
        
        entry_price = prices_df.loc[entry_date, ticker]
        exit_price = prices_df.loc[exit_date, ticker]
        
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
            continue
        
        raw_return = ((exit_price / entry_price) - 1) * 100
        
        # Direction: long for beats, short for misses
        direction = 1 if category == "Beat" else -1
        trade_return = raw_return * direction
        
        # SPY benchmark return over same period
        spy_return = np.nan
        if config.BENCHMARK_TICKER in prices_df.columns:
            spy_entry = prices_df.loc[entry_date, config.BENCHMARK_TICKER]
            spy_exit = prices_df.loc[exit_date, config.BENCHMARK_TICKER]
            if not pd.isna(spy_entry) and not pd.isna(spy_exit) and spy_entry != 0:
                spy_return = ((spy_exit / spy_entry) - 1) * 100
        
        trades.append({
            "ticker": ticker,
            "category": category,
            "direction": "Long" if direction == 1 else "Short",
            "earnings_date": earn_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_return_pct": raw_return,
            "trade_return_pct": trade_return,
            "spy_return_pct": spy_return,
            "excess_return_pct": trade_return - spy_return if not pd.isna(spy_return) else np.nan,
            "surprise_pct": event["surprise_pct"],
        })
    
    if not trades:
        if verbose:
            print("  ✗ No valid trades could be generated.")
        return pd.DataFrame(), {}
    
    trade_log = pd.DataFrame(trades)
    trade_log = trade_log.sort_values("entry_date").reset_index(drop=True)

    # ── Compute Performance Metrics ─────────────────────────────────────
    returns = trade_log["trade_return_pct"]
    excess = trade_log["excess_return_pct"].dropna()

    # Group trades that share an entry date into an equal-weighted portfolio
    # for that date — capital cannot compound across simultaneous trades, so
    # each batch contributes the mean of its constituent returns.
    batch = trade_log.groupby("entry_date").agg(
        batch_return_pct=("trade_return_pct", "mean"),
        batch_spy_pct=("spy_return_pct", "mean"),
        batch_size=("trade_return_pct", "size"),
    ).sort_index()

    batch_cum = (1 + batch["batch_return_pct"] / 100).cumprod()
    total_return = (batch_cum.iloc[-1] - 1) * 100

    # Sharpe ratio on batch (portfolio) returns, annualized by batches/year
    if batch["batch_return_pct"].std() > 0:
        date_span_days = max((batch.index.max() - batch.index.min()).days, 1)
        batches_per_year = len(batch) * 365.0 / date_span_days
        sharpe = (batch["batch_return_pct"].mean() /
                  batch["batch_return_pct"].std()) * np.sqrt(batches_per_year)
    else:
        sharpe = 0

    # Max drawdown on the portfolio equity curve
    cum_max = batch_cum.cummax()
    drawdown = (batch_cum - cum_max) / cum_max * 100
    max_dd = drawdown.min()

    # Win rate (per individual trade, more informative than per batch)
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    # By category
    long_trades = trade_log[trade_log["direction"] == "Long"]["trade_return_pct"]
    short_trades = trade_log[trade_log["direction"] == "Short"]["trade_return_pct"]
    
    metrics = {
        "total_trades": len(trades),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "total_return_pct": total_return,
        "avg_trade_return_pct": returns.mean(),
        "median_trade_return_pct": returns.median(),
        "std_trade_return_pct": returns.std(),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "win_rate_pct": win_rate,
        "avg_long_return_pct": long_trades.mean() if len(long_trades) > 0 else np.nan,
        "avg_short_return_pct": short_trades.mean() if len(short_trades) > 0 else np.nan,
        "avg_excess_return_pct": excess.mean() if len(excess) > 0 else np.nan,
        "best_trade_pct": returns.max(),
        "worst_trade_pct": returns.min(),
    }
    
    # ── Build equity curve for plotting ─────────────────────────────────
    # The equity curve lives at the batch (entry-date) level, not per-trade,
    # so simultaneous trades show as one step rather than a vertical stack.
    equity_df = pd.DataFrame({
        "date": batch.index,
        "cumulative_return": (batch_cum.values - 1) * 100,
        "batch_return_pct": batch["batch_return_pct"].values,
        "batch_size": batch["batch_size"].values,
    }).reset_index(drop=True)

    spy_batch_cum = (1 + batch["batch_spy_pct"].fillna(0) / 100).cumprod()
    equity_df["benchmark_cumulative"] = (spy_batch_cum.values - 1) * 100

    # Keep per-trade returns alongside the equity curve so the lower
    # panel of the chart can show individual trade outcomes.
    trade_log["trade_return"] = trade_log["trade_return_pct"]
    trade_log["date"] = trade_log["entry_date"]
    
    if verbose:
        print("\n" + "="*60)
        print("  BACKTEST RESULTS")
        print("="*60)
        print(f"\n  Strategy: Long Beats / Short Misses (hold {hold_days} days)")
        print(f"  {'─'*50}")
        print(f"  Total Trades:        {metrics['total_trades']}")
        print(f"    Long (Beats):      {metrics['long_trades']}")
        print(f"    Short (Misses):    {metrics['short_trades']}")
        print(f"  {'─'*50}")
        print(f"  Total Return:        {metrics['total_return_pct']:+.2f}%")
        print(f"  Avg Trade Return:    {metrics['avg_trade_return_pct']:+.3f}%")
        print(f"  Median Trade Return: {metrics['median_trade_return_pct']:+.3f}%")
        print(f"  Std Dev:             {metrics['std_trade_return_pct']:.3f}%")
        print(f"  {'─'*50}")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
        print(f"  Win Rate:            {metrics['win_rate_pct']:.1f}%")
        print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"  {'─'*50}")
        print(f"  Avg Long Return:     {metrics['avg_long_return_pct']:+.3f}%")
        print(f"  Avg Short Return:    {metrics['avg_short_return_pct']:+.3f}%")
        print(f"  Avg Excess Return:   {metrics['avg_excess_return_pct']:+.3f}%")
        print(f"  {'─'*50}")
        print(f"  Best Trade:          {metrics['best_trade_pct']:+.2f}%")
        print(f"  Worst Trade:         {metrics['worst_trade_pct']:+.2f}%")

    # Attach the portfolio-level cumulative return at each trade's entry date
    # so per-trade rows still carry the equity curve value the visualizer needs.
    eq_lookup = equity_df.set_index("date")
    trade_log["cumulative_return"] = trade_log["entry_date"].map(
        eq_lookup["cumulative_return"]
    )
    trade_log["benchmark_cumulative"] = trade_log["entry_date"].map(
        eq_lookup["benchmark_cumulative"]
    )

    # Stash the deduplicated equity curve on the DataFrame as an attribute
    # so the visualizer can plot one point per entry date (no vertical stacks).
    trade_log.attrs["equity_curve"] = equity_df

    return trade_log, metrics
