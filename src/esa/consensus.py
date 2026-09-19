"""Analyst consensus, and what is wrong with the only free source of it.

The study measures surprise against the firm's own past, not against analyst
expectations, because EDGAR does not publish consensus and the free
alternatives are not point-in-time. This module exists to make that claim
checkable rather than asserted, by fetching the consensus figures anyway and
running the headline test both ways.

**What the field actually is.** ``yfinance.Ticker.get_earnings_dates`` reads
Yahoo Finance's earnings calendar and returns three columns: ``EPS Estimate``,
``Reported EPS`` and ``Surprise(%)``. The estimate is the consensus Yahoo
currently publishes for that quarter, served from the same endpoint whether
the quarter is next week or eight years ago. Nothing in the response records
when that consensus was formed or whether it has been revised since.

Two consequences follow, and both are visible in the data rather than
theoretical:

*Revision contamination.* If a broker updated a stale estimate after the
announcement, the difference being measured is partly built from information
that did not exist when the stock moved.

*Composition drift.* The set of brokers contributing to a consensus changes.
The figure served today for a 2015 quarter need not be the figure anyone saw
in 2015.

A paid point-in-time source — I/B/E/S, Zacks, Refinitiv — timestamps each
estimate revision and can reconstruct the consensus as it stood the evening
before a release. None is free, and none is used here.

There is also a units trap in this field worth recording, because the previous
version of this repository fell into it: ``Surprise(%)`` is already a
percentage. Reading it as a fraction and multiplying by a hundred turns a 6.7%
surprise into 674%.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import config

#: Yahoo serves a bounded window of quarters per company, so asking for more
#: than this returns nothing extra.
MAX_QUARTERS = 40

#: An announcement in this study and a calendar entry at Yahoo can disagree by
#: a day either way, usually because one is recorded in local time and the
#: other in Eastern.
MATCH_TOLERANCE_DAYS = 2


def fetch_consensus(
    tickers: list[str], *, pause_s: float = 0.2, verbose: bool = True
) -> pd.DataFrame:
    """Download Yahoo's consensus estimates and reported EPS per quarter.

    Returns ``[ticker, announced_date, eps_estimate, eps_actual,
    yahoo_surprise_pct]``, one row per past quarter Yahoo has a figure for.
    """
    import yfinance as yf

    rows: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        if verbose and i % 50 == 0:
            print(f"  consensus {i}/{len(tickers)}", flush=True)
        try:
            frame = yf.Ticker(ticker).get_earnings_dates(limit=MAX_QUARTERS)
        except Exception:
            # A company Yahoo has no calendar for is a gap in an optional
            # cross-check, not a reason to abandon five hundred others.
            continue
        if frame is None or frame.empty:
            continue
        frame = frame.dropna(subset=["Reported EPS", "EPS Estimate"])
        if frame.empty:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "announced_date": pd.DatetimeIndex(frame.index)
                    .tz_convert("America/New_York")
                    .tz_localize(None)
                    .normalize(),
                    "eps_estimate": frame["EPS Estimate"].to_numpy(dtype=float),
                    "eps_actual": frame["Reported EPS"].to_numpy(dtype=float),
                    "yahoo_surprise_pct": frame.get("Surprise(%)", pd.Series(dtype=float)).to_numpy(
                        dtype=float
                    )
                    if "Surprise(%)" in frame.columns
                    else float("nan"),
                }
            )
        )
        time.sleep(pause_s)

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "announced_date",
                "eps_estimate",
                "eps_actual",
                "yahoo_surprise_pct",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["ticker", "announced_date"]).reset_index(drop=True)


def save_consensus(frame: pd.DataFrame, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or config.CACHE_DIR / "consensus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "yahoo_consensus.parquet"
    frame.to_parquet(path)
    config.write_provenance(
        path,
        command="python scripts/fetch_consensus.py",
        data_source=(
            "Yahoo Finance earnings calendar via yfinance Ticker.get_earnings_dates; "
            "a current snapshot of consensus, not a point-in-time one"
        ),
        as_of=config.StudyConfig().end_date,
        extra={"n_rows": len(frame), "n_tickers": int(frame["ticker"].nunique())},
    )
    return path


def load_consensus(cache_dir: Path | None = None) -> pd.DataFrame:
    cache_dir = cache_dir or config.CACHE_DIR / "consensus"
    path = cache_dir / "yahoo_consensus.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `python scripts/fetch_consensus.py`")
    return pd.read_parquet(path)


def attach_to_events(events: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """Join consensus surprise onto the event panel by ticker and date.

    Matched within :data:`MATCH_TOLERANCE_DAYS`, per ticker, nearest first.
    Events with no consensus row are kept with NaN so the caller can count the
    overlap rather than silently studying a subsample.
    """
    from .surprise import percent_surprise

    if consensus.empty:
        return events.assign(pct_surprise=float("nan"))

    right = consensus.copy()
    right["pct_surprise"] = [
        percent_surprise(a, e) for a, e in zip(right["eps_actual"], right["eps_estimate"])
    ]
    right = right.sort_values("announced_date")

    left = events.sort_values("announcement_date")
    merged = pd.merge_asof(
        left,
        right[["ticker", "announced_date", "eps_estimate", "eps_actual", "pct_surprise"]],
        left_on="announcement_date",
        right_on="announced_date",
        by="ticker",
        direction="nearest",
        tolerance=pd.Timedelta(days=MATCH_TOLERANCE_DAYS),
    )
    return merged.reset_index(drop=True)
