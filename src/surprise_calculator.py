"""
Surprise Calculator Module
Computes earnings surprise categories and post-announcement stock returns.
"""

import numpy as np
import pandas as pd

import config


def categorize_surprise(surprise_pct: float) -> str:
    """Categorize an earnings surprise as Beat, Miss, or In-Line."""
    if surprise_pct > config.BEAT_THRESHOLD:
        return "Beat"
    elif surprise_pct < config.MISS_THRESHOLD:
        return "Miss"
    else:
        return "In-Line"


def compute_forward_returns(
    earnings_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    windows: list[int] = None,
) -> pd.DataFrame:
    """
    For each earnings event, compute forward stock returns over various windows.
    
    Also computes market-adjusted returns (raw return - SPY return).
    
    Args:
        earnings_df: DataFrame with columns [ticker, earnings_date, ...]
        prices_df: DataFrame with DatetimeIndex, columns = ticker symbols
        windows: List of forward return windows in trading days
    
    Returns:
        earnings_df augmented with return columns:
          ret_day{N}      — raw cumulative return
          ret_day{N}_adj  — market-adjusted return
    """
    windows = windows or config.RETURN_WINDOWS
    
    # Ensure dates are compatible
    prices_df.index = pd.to_datetime(prices_df.index).normalize()
    trading_dates = prices_df.index.sort_values()
    
    # Pre-build a date → position map for fast lookup
    date_to_pos = {d: i for i, d in enumerate(trading_dates)}
    
    results = []
    
    for _, row in earnings_df.iterrows():
        ticker = row["ticker"]
        earn_date = pd.Timestamp(row["earnings_date"]).normalize()
        
        # Skip if ticker not in price data
        if ticker not in prices_df.columns:
            continue
        
        # Find the closest trading day on or after the earnings date
        # (earnings may be announced on a non-trading day)
        matching_dates = trading_dates[trading_dates >= earn_date]
        if len(matching_dates) == 0:
            continue
        
        event_date = matching_dates[0]
        event_pos = date_to_pos[event_date]
        
        # We need the price on the day BEFORE the event to compute day-0 return
        if event_pos == 0:
            continue
        
        record = row.to_dict()
        record["event_trading_date"] = event_date
        
        ticker_prices = prices_df[ticker].values
        spy_prices = prices_df[config.BENCHMARK_TICKER].values if config.BENCHMARK_TICKER in prices_df.columns else None
        
        # Pre-event close (for day-0 return calculation)
        pre_event_price = ticker_prices[event_pos - 1]
        pre_event_spy = spy_prices[event_pos - 1] if spy_prices is not None else None
        
        valid = True
        for w in windows:
            target_pos = event_pos + w
            
            if target_pos >= len(trading_dates) or target_pos < 0:
                record[f"ret_day{w}"] = np.nan
                record[f"ret_day{w}_adj"] = np.nan
                continue
            
            if w == 0:
                # Day-0 return: from previous close to event-day close
                base_price = pre_event_price
                base_spy = pre_event_spy
            else:
                # Day-N return: from event-day close to day-N close
                base_price = ticker_prices[event_pos]
                base_spy = spy_prices[event_pos] if spy_prices is not None else None
            
            end_price = ticker_prices[target_pos]
            
            if pd.isna(base_price) or pd.isna(end_price) or base_price == 0:
                record[f"ret_day{w}"] = np.nan
                record[f"ret_day{w}_adj"] = np.nan
                continue
            
            raw_ret = ((end_price / base_price) - 1) * 100  # percentage
            record[f"ret_day{w}"] = raw_ret
            
            # Market-adjusted return
            if base_spy is not None and spy_prices is not None:
                end_spy = spy_prices[target_pos]
                if not pd.isna(base_spy) and not pd.isna(end_spy) and base_spy != 0:
                    spy_ret = ((end_spy / base_spy) - 1) * 100
                    record[f"ret_day{w}_adj"] = raw_ret - spy_ret
                else:
                    record[f"ret_day{w}_adj"] = np.nan
            else:
                record[f"ret_day{w}_adj"] = np.nan
        
        results.append(record)
    
    if not results:
        raise ValueError("No valid earnings events with matching price data found.")
    
    result_df = pd.DataFrame(results)
    
    return result_df


def build_analysis_dataset(
    earnings_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: categorize surprises, compute returns, merge everything.
    
    Returns:
        DataFrame with columns:
        [ticker, earnings_date, eps_estimate, eps_actual, surprise_pct,
         category, ret_day0, ret_day5, ret_day20, ret_day0_adj, ...]
    """
    if verbose:
        print(f"  Computing forward returns for {len(earnings_df)} earnings events...")
    
    # Compute forward returns
    merged = compute_forward_returns(earnings_df, prices_df)
    
    # Categorize surprises
    merged["category"] = merged["surprise_pct"].apply(categorize_surprise)
    
    # Sort by date
    merged = merged.sort_values("earnings_date").reset_index(drop=True)
    
    if verbose:
        cat_counts = merged["category"].value_counts()
        print(f"  ✓ {len(merged)} events with returns computed")
        print(f"    Beats: {cat_counts.get('Beat', 0)} | "
              f"Misses: {cat_counts.get('Miss', 0)} | "
              f"In-Line: {cat_counts.get('In-Line', 0)}")
    
    return merged
