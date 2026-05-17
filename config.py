"""
Configuration for the Earnings Surprise Analyzer.
Contains stock universe, thresholds, and paths.
"""

import os
from datetime import datetime, timedelta

# ─── Stock Universe ──────────────────────────────────────────────────────────
# 25 S&P 500 mega-cap stocks across sectors
TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "NFLX", "DIS",
    # Financials
    "JPM", "BAC", "GS",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Energy
    "XOM", "CVX",
    # Consumer Staples
    "PG", "KO", "WMT", "COST",
    # Semiconductors
    "AMD", "INTC",
    # Software
    "CRM",
]

BENCHMARK_TICKER = "SPY"

# ─── Earnings Surprise Thresholds ────────────────────────────────────────────
# Surprise magnitude thresholds (in percentage points)
# Most mega-cap stocks beat/miss by 3-15%, so 5% provides good separation
BEAT_THRESHOLD = 5.0      # surprise > +5%  → "Beat"
MISS_THRESHOLD = -5.0     # surprise < -5%  → "Miss"
                          # otherwise        → "In-Line"

# ─── Return Windows (trading days) ───────────────────────────────────────────
RETURN_WINDOWS = [0, 1, 5, 10, 20]  # Day-0, Day-1, Day-5, Day-10, Day-20

# ─── Date Range for Price Data ───────────────────────────────────────────────
PRICE_START_DATE = "2023-01-01"
PRICE_END_DATE = datetime.now().strftime("%Y-%m-%d")

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

# Cache files
EARNINGS_CACHE = os.path.join(DATA_DIR, "earnings_raw.csv")
PRICES_CACHE = os.path.join(DATA_DIR, "prices.csv")
MERGED_CACHE = os.path.join(DATA_DIR, "earnings_with_returns.csv")
STATS_OUTPUT = os.path.join(DATA_DIR, "statistical_results.csv")
BACKTEST_OUTPUT = os.path.join(DATA_DIR, "backtest_results.csv")

# ─── Visualization ───────────────────────────────────────────────────────────
CHART_STYLE = "dark_background"
CHART_DPI = 150
CHART_FIGSIZE = (12, 7)

# Color palette
COLORS = {
    "beat": "#00E676",       # Vibrant green
    "miss": "#FF5252",       # Vibrant red
    "inline": "#FFD740",     # Amber
    "accent": "#40C4FF",     # Light blue
    "bg_dark": "#1a1a2e",    # Dark navy
    "bg_card": "#16213e",    # Card background
    "text": "#e0e0e0",       # Light text
    "grid": "#333355",       # Grid lines
}

# ─── Backtest Parameters ─────────────────────────────────────────────────────
BACKTEST_HOLD_DAYS = 5        # Hold position for 5 trading days
BACKTEST_CAPITAL = 100_000    # Starting capital
BACKTEST_POSITION_SIZE = 0.1  # 10% of capital per trade


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
