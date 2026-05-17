"""
Data Fetcher Module
Fetches earnings history and price data from Yahoo Finance via yfinance.
Implements caching to avoid redundant API calls.
"""

import os
import time
import warnings

import pandas as pd
import yfinance as yf

import config


def fetch_earnings_data(tickers: list[str], verbose: bool = True) -> pd.DataFrame:
    """
    Fetch earnings history (EPS estimate, actual, surprise) for a list of tickers.
    
    Uses yfinance's earnings_dates attribute which provides:
    - EPS Estimate
    - Reported EPS  
    - Surprise(%)
    
    Falls back to earnings_history if earnings_dates is unavailable.
    
    Returns:
        DataFrame with columns: [ticker, earnings_date, eps_estimate, eps_actual, surprise_pct]
    """
    all_earnings = []
    
    for i, ticker in enumerate(tickers):
        if verbose:
            print(f"  [{i+1}/{len(tickers)}] Fetching earnings for {ticker}...", end=" ")
        
        try:
            t = yf.Ticker(ticker)
            
            # ── Try earnings_dates first (more data) ────────────────────────
            df = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = t.get_earnings_dates(limit=20)
            except Exception:
                pass
            
            if df is not None and not df.empty:
                # Filter to rows that have actual reported EPS (past earnings only)
                df = df.dropna(subset=["Reported EPS"])
                
                if not df.empty:
                    records = []
                    for date_idx, row in df.iterrows():
                        eps_est = row.get("EPS Estimate")
                        eps_act = row.get("Reported EPS")
                        surprise = row.get("Surprise(%)")
                        
                        # Skip if no estimate available
                        if pd.isna(eps_est) or pd.isna(eps_act):
                            continue
                        
                        # Calculate surprise if not provided
                        if pd.isna(surprise):
                            if abs(eps_est) > 0.001:
                                surprise = ((eps_act - eps_est) / abs(eps_est)) * 100
                            else:
                                surprise = 0.0
                        else:
                            # yfinance returns Surprise(%) as a fraction
                            # (e.g., 0.10 = 10%), so convert to percentage
                            surprise = surprise * 100
                        
                        # Cap extreme surprise values from near-zero EPS estimates
                        surprise = max(min(float(surprise), 200.0), -200.0)
                        
                        records.append({
                            "ticker": ticker,
                            "earnings_date": pd.Timestamp(date_idx).tz_localize(None).normalize(),
                            "eps_estimate": float(eps_est),
                            "eps_actual": float(eps_act),
                            "surprise_pct": surprise,
                        })
                    
                    if records:
                        all_earnings.extend(records)
                        if verbose:
                            print(f"✓ {len(records)} earnings events")
                        time.sleep(0.3)  # Rate limiting
                        continue
            
            # ── Fallback: try earnings_history ──────────────────────────────
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eh = t.earnings_history
                
                if eh is not None and not eh.empty:
                    records = []
                    for date_idx, row in eh.iterrows():
                        eps_est = row.get("epsEstimate", row.get("EPS Estimate"))
                        eps_act = row.get("epsActual", row.get("Reported EPS"))
                        surprise = row.get("surprisePercent", row.get("Surprise(%)"))
                        
                        if pd.isna(eps_est) or pd.isna(eps_act):
                            continue
                        
                        if pd.isna(surprise):
                            if abs(eps_est) > 0.001:
                                surprise = ((eps_act - eps_est) / abs(eps_est)) * 100
                            else:
                                surprise = 0.0
                        else:
                            # Convert fraction to percentage
                            surprise = surprise * 100
                        
                        # Cap extreme surprise values
                        surprise = max(min(float(surprise), 200.0), -200.0)
                        
                        records.append({
                            "ticker": ticker,
                            "earnings_date": pd.Timestamp(date_idx).tz_localize(None).normalize(),
                            "eps_estimate": float(eps_est),
                            "eps_actual": float(eps_act),
                            "surprise_pct": surprise,
                        })
                    
                    if records:
                        all_earnings.extend(records)
                        if verbose:
                            print(f"✓ {len(records)} events (fallback)")
                        time.sleep(0.3)
                        continue
            except Exception:
                pass
            
            if verbose:
                print("✗ No data available")
                
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
        
        time.sleep(0.3)  # Rate limiting
    
    if not all_earnings:
        raise ValueError("No earnings data could be fetched for any ticker.")
    
    df_earnings = pd.DataFrame(all_earnings)
    
    # Deduplicate by (ticker, earnings_date)
    df_earnings = df_earnings.drop_duplicates(subset=["ticker", "earnings_date"])
    df_earnings = df_earnings.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    
    return df_earnings


def fetch_price_data(
    tickers: list[str],
    start: str = None,
    end: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices for all tickers + benchmark.
    
    Returns:
        DataFrame with DatetimeIndex and one column per ticker.
    """
    start = start or config.PRICE_START_DATE
    end = end or config.PRICE_END_DATE
    
    all_tickers = list(set(tickers + [config.BENCHMARK_TICKER]))
    
    if verbose:
        print(f"  Downloading prices for {len(all_tickers)} tickers ({start} → {end})...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            all_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    
    # yf.download returns MultiIndex columns for multiple tickers: (Price, Ticker)
    # We want just the Close prices
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    
    # Ensure column names are strings
    df.columns = [str(c) for c in df.columns]
    
    # Forward-fill then backward-fill missing prices
    df = df.ffill().bfill()
    
    # Ensure index is timezone-naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    if verbose:
        print(f"  ✓ Got {len(df)} trading days of price data")
    
    return df


def load_or_fetch_earnings(
    tickers: list[str],
    cache_path: str = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load earnings from cache or fetch fresh."""
    cache_path = cache_path or config.EARNINGS_CACHE
    
    if use_cache and os.path.exists(cache_path):
        if verbose:
            print(f"  Loading cached earnings from {os.path.basename(cache_path)}")
        df = pd.read_csv(cache_path, parse_dates=["earnings_date"])
        return df
    
    df = fetch_earnings_data(tickers, verbose=verbose)
    
    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    if verbose:
        print(f"  Cached earnings data → {os.path.basename(cache_path)}")
    
    return df


def load_or_fetch_prices(
    tickers: list[str],
    cache_path: str = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load prices from cache or fetch fresh."""
    cache_path = cache_path or config.PRICES_CACHE
    
    if use_cache and os.path.exists(cache_path):
        if verbose:
            print(f"  Loading cached prices from {os.path.basename(cache_path)}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df
    
    df = fetch_price_data(tickers, verbose=verbose)
    
    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path)
    if verbose:
        print(f"  Cached price data → {os.path.basename(cache_path)}")
    
    return df
