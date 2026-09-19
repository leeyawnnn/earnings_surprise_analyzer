"""Wiring: filings and prices in, an event panel and a results set out.

The joins in :func:`build_events` are where a study of this kind either stays
honest or quietly stops being one, so each one carries the rule it enforces.
Three guards run here and nowhere else:

1. A quarter is matched to the *first* 8-K carrying item 2.02 that follows the
   period end, so the event date is the day the result became public rather
   than the day the quarter ended.
2. An event survives only if every input to its surprise measure was already
   on file when the announcement landed.
3. A company enters the sample only from the date it joined the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, factors, universe
from .config import StudyConfig
from .events import build_event_panel
from .prices import PricePanel, download_prices, load_panel, save_panel
from .sec import SecClient, quarterly_eps
from .surprise import categorize_series, standardized_unexpected_earnings

#: A result is announced within a few months of the period it covers. Anything
#: further out is a mismatch, usually a company whose 8-K history has a gap.
MAX_ANNOUNCEMENT_LAG_DAYS = 100
#: And never before the books close.
MIN_ANNOUNCEMENT_LAG_DAYS = 1

#: A quarter's earnings per share cannot plausibly exceed a quarter of the
#: share price: that would be a trailing price/earnings ratio near one. The
#: filter exists because a small number of filers tag this concept wrongly.
MAX_QUARTERLY_EPS_YIELD = 0.25


@dataclass
class RawData:
    """Everything the study reads, after downloading and before joining."""

    announcements: pd.DataFrame
    eps: pd.DataFrame
    panel: PricePanel
    factors: pd.DataFrame
    universe: pd.DataFrame


def fetch_all(cfg: StudyConfig, *, verbose: bool = True) -> None:
    """Download filings, prices and factors into ``data/cache``."""
    config.ensure_dirs()
    members = universe.load_universe()
    client = SecClient()

    announcements: list[pd.DataFrame] = []
    eps_frames: list[pd.DataFrame] = []
    for i, row in enumerate(members.itertuples()):
        if verbose and i % 50 == 0:
            print(f"  EDGAR {i}/{len(members)}", flush=True)
        cik = int(row.cik)
        announced = client.earnings_announcements(cik)
        if not announced.empty:
            announcements.append(announced.assign(ticker=row.ticker))
        reported = quarterly_eps(client.eps_facts(cik))
        if not reported.empty:
            eps_frames.append(reported.assign(ticker=row.ticker))

    sec_dir = config.CACHE_DIR / "sec"
    sec_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(announcements, ignore_index=True).to_parquet(sec_dir / "announcements.parquet")
    pd.concat(eps_frames, ignore_index=True).to_parquet(sec_dir / "quarterly_eps.parquet")

    tickers = sorted({*members["ticker"], config.BENCHMARK_TICKER})
    panel = download_prices(tickers, cfg.start_date, cfg.end_date, verbose=verbose)
    save_panel(panel, config.CACHE_DIR / "prices", source_as_of=cfg.end_date)
    factors.download_factors()


def load_raw(cache_dir: Path | None = None) -> RawData:
    """Read everything :func:`fetch_all` wrote.

    ``cache_dir`` points at ``data/sample`` for the CI run, which exercises the
    whole pipeline on a committed slice small enough to keep in the repo.
    """
    cache_dir = cache_dir or config.CACHE_DIR
    sec_dir = cache_dir / "sec"
    universe_path = cache_dir / "universe.csv"
    return RawData(
        announcements=pd.read_parquet(sec_dir / "announcements.parquet"),
        eps=pd.read_parquet(sec_dir / "quarterly_eps.parquet"),
        panel=load_panel(cache_dir / "prices"),
        factors=factors.load_factors(cache_dir / "factors"),
        universe=universe.load_universe(universe_path if universe_path.exists() else None),
    )


def match_announcements(eps: pd.DataFrame, announcements: pd.DataFrame) -> pd.DataFrame:
    """Attach each fiscal quarter to the filing that announced it.

    The match is the first item-2.02 8-K accepted after the period ends. It is
    done per ticker with a forward as-of join, and a quarter with no filing
    inside the tolerance is dropped rather than matched to a later quarter's
    release.
    """
    left = eps.sort_values("end").copy()
    left["_key"] = left["end"] + pd.Timedelta(days=MIN_ANNOUNCEMENT_LAG_DAYS)

    right = announcements.copy()
    right["accepted_utc"] = pd.to_datetime(right["accepted_utc"], utc=True)
    right["_key"] = right["accepted_utc"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    right = right.sort_values("_key")

    merged = pd.merge_asof(
        left.sort_values("_key"),
        right[["ticker", "accession", "accepted_utc", "_key"]].sort_values("_key"),
        on="_key",
        by="ticker",
        direction="forward",
        tolerance=pd.Timedelta(days=MAX_ANNOUNCEMENT_LAG_DAYS),
    )
    merged = merged.dropna(subset=["accepted_utc"]).drop(columns="_key")

    # One filing announces one quarter. If two quarters point at the same 8-K,
    # the earlier quarter is the stale one and is dropped.
    merged = merged.sort_values(["ticker", "end"]).drop_duplicates(
        ["ticker", "accession"], keep="last"
    )
    return merged.reset_index(drop=True)


def plausible_eps_mask(
    events: pd.DataFrame, panel: PricePanel, *, max_yield: float = MAX_QUARTERLY_EPS_YIELD
) -> pd.Series:
    """Drop firm-quarters whose tagged EPS cannot be a per-share figure.

    Two failure modes show up in XBRL and both are loud rather than subtle.
    A few filers tag ``EarningsPerShareDiluted`` with a whole-dollar income
    figure — Intercontinental Exchange reports 120,000,000 for a 2016
    quarter. And a dual-class issuer tags the concept for its Class A shares
    while the traded, index-member line is the Class B: Berkshire's Class A
    earnings per share are three orders of magnitude above the Class B price.

    A flat dollar cap would not separate these from genuinely high-priced
    names — NVR earns well over a hundred dollars a share in a quarter — so
    the test is relative to the share price at the period end. A quarterly
    earnings yield above 25% implies a trailing price/earnings ratio of about
    one, which no index constituent has.
    """
    closes = panel.close
    positions = closes.index.searchsorted(events["end"].to_numpy(), side="right") - 1
    positions = np.clip(positions, 0, len(closes) - 1)
    col_of = {t: i for i, t in enumerate(closes.columns)}
    cols = events["ticker"].map(col_of)
    known = cols.notna()
    price = np.full(len(events), np.nan)
    price[known.to_numpy()] = closes.to_numpy()[
        positions[known.to_numpy()], cols[known].astype(int).to_numpy()
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.abs(events["val"].to_numpy()) / price
    # An unknown price is not evidence of a bad tag, so it passes here and is
    # removed later by the price-window filter.
    return pd.Series(~(ratio > max_yield), index=events.index).fillna(True)


def build_events(
    raw: RawData, cfg: StudyConfig, *, verbose: bool = True
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assemble the event panel and a record of what each filter removed."""
    audit: dict[str, int] = {}

    eps = raw.eps.copy()
    eps["end"] = pd.to_datetime(eps["end"])
    eps["filed"] = pd.to_datetime(eps["filed"])
    audit["quarterly_eps_rows"] = len(eps)

    sue = standardized_unexpected_earnings(eps, history_quarters=cfg.sue_history_quarters)
    audit["with_sue"] = int(sue["sue"].notna().sum())

    matched = match_announcements(sue, raw.announcements)
    audit["matched_to_8k"] = len(matched)

    matched = matched[plausible_eps_mask(matched, raw.panel)]
    audit["after_eps_plausibility"] = len(matched)

    matched = matched.dropna(subset=["sue"])
    audit["after_sue_required"] = len(matched)

    # Guard 2: the whole forecast must have been on file before the release.
    announced_day = (
        matched["accepted_utc"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    known = matched["history_known_by"].notna() & (matched["history_known_by"] < announced_day)
    matched = matched[known]
    audit["after_point_in_time_guard"] = len(matched)

    matched["announcement_date"] = announced_day[matched.index]
    eligible = universe.eligibility_mask(raw.universe, matched)
    matched = matched[eligible]
    audit["after_index_membership"] = len(matched)

    window = (matched["announcement_date"] >= pd.Timestamp(cfg.event_start_date)) & (
        matched["announcement_date"] <= pd.Timestamp(cfg.end_date)
    )
    matched = matched[window]
    audit["in_study_window"] = len(matched)

    panel = build_event_panel(matched, raw.panel, windows=cfg.windows)
    audit["with_price_windows"] = len(panel)

    longest = max(cfg.windows)
    panel = panel.dropna(subset=[f"ret_d{longest}", "gap_ret", "open_to_close_ret"])
    audit["with_complete_returns"] = len(panel)

    panel["category"] = categorize_series(
        panel["sue"], beat=cfg.beat_threshold, miss=cfg.miss_threshold
    )
    panel["fiscal_quarter_end"] = panel["end"]
    if verbose:
        for key, value in audit.items():
            print(f"  {key:<28} {value:>8,}")
    return panel.reset_index(drop=True), audit


def save_events(panel: pd.DataFrame, cfg: StudyConfig, out_dir: Path | None = None) -> Path:
    """Write the event panel and its provenance sidecar."""
    config.ensure_dirs()
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "event_panel.parquet"
    panel.to_parquet(path)
    config.write_provenance(
        path,
        command="python main.py study",
        data_source=(
            "SEC EDGAR 8-K item 2.02 and XBRL us-gaap:EarningsPerShareDiluted; Yahoo Finance prices"
        ),
        as_of=cfg.end_date,
        extra={
            "n_events": len(panel),
            "n_tickers": int(panel["ticker"].nunique()),
            "first_event": str(panel["announcement_date"].min().date()),
            "last_event": str(panel["announcement_date"].max().date()),
            "sue_history_quarters": cfg.sue_history_quarters,
            "beat_threshold_sigma": cfg.beat_threshold,
            "miss_threshold_sigma": cfg.miss_threshold,
        },
    )
    return path


def load_events() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "event_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `python main.py study` first")
    return pd.read_parquet(path)


def timing_breakdown(panel: pd.DataFrame) -> pd.DataFrame:
    """How announcements split between before-open, intraday and after-close."""
    counts = panel["timing"].value_counts()
    return pd.DataFrame(
        {
            "timing": counts.index,
            "n": counts.to_numpy(),
            "share_pct": np.round(counts.to_numpy() / len(panel) * 100.0, 2),
        }
    ).reset_index(drop=True)
