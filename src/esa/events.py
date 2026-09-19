"""Turning filings into an event panel.

The question this module answers is narrow and easy to get wrong: given that a
company released results at a particular moment, which session is the first
one a trader could have acted in, and what happened afterwards?

EDGAR records the acceptance time of the 8-K to the second, so the answer does
not have to be assumed. A release at 07:00 Eastern is tradeable from that
morning's open; a release at 16:30 is not tradeable until the next morning.
Getting this wrong by one session is the difference between measuring drift
and measuring the announcement reaction itself.

Every price this module reads is checked against the announcement timestamp by
:func:`assert_no_lookahead`, which is the guard the test suite pins.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from . import config
from .prices import PricePanel

EASTERN = "America/New_York"
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

BEFORE_OPEN = "before_open"
INTRADAY = "intraday"
AFTER_CLOSE = "after_close"


def classify_timing(accepted_utc: pd.Series | pd.Timestamp) -> pd.Series | str:
    """Label an acceptance timestamp as before-open, intraday or after-close.

    EDGAR stamps acceptance in UTC, so the conversion to Eastern has to happen
    before the comparison — an Apple release at 20:30 UTC is 16:30 in New York
    and is after-close, not mid-afternoon.
    """
    if isinstance(accepted_utc, pd.Timestamp):
        local = accepted_utc.tz_convert(EASTERN)
        return _label(local.time())

    local = pd.to_datetime(accepted_utc, utc=True).dt.tz_convert(EASTERN)
    clock = local.dt.time
    return pd.Series([_label(t) for t in clock], index=accepted_utc.index, dtype="object")


def _label(clock: time) -> str:
    if clock < MARKET_OPEN:
        return BEFORE_OPEN
    if clock <= MARKET_CLOSE:
        return INTRADAY
    return AFTER_CLOSE


def reaction_positions(
    announced_et: pd.Series, timing: pd.Series, panel: PricePanel
) -> np.ndarray:
    """Index of the first session in which the market can react to each release.

    A release before the opening bell is absorbed by that same session. One
    after the close, or after hours on a non-trading day, is absorbed by the
    next session. Positions past the end of the price calendar come back as
    ``len(dates)`` and are filtered out by the caller.
    """
    dates = panel.dates
    announced_day = pd.DatetimeIndex(announced_et.dt.normalize().to_numpy())
    pos = dates.searchsorted(announced_day, side="left").astype(int)

    is_after_close = (timing == AFTER_CLOSE).to_numpy()
    on_a_session = np.zeros(len(pos), dtype=bool)
    in_range = pos < len(dates)
    on_a_session[in_range] = dates[pos[in_range]].to_numpy() == announced_day[in_range].to_numpy()
    pos = pos + (is_after_close & on_a_session).astype(int)
    return pos


def entry_position_column(entry_timing: str) -> str:
    """Which precomputed entry column a fill convention uses."""
    if entry_timing == "next_open":
        return "entry_pos_open"
    if entry_timing == "next_close":
        return "entry_pos_close"
    raise ValueError(f"unknown entry_timing: {entry_timing!r}")


def entry_timestamp(session: pd.Timestamp, entry_timing: str) -> pd.Timestamp:
    """UTC instant at which an order fills, given the session and fill type."""
    clock = MARKET_OPEN if entry_timing == "next_open" else MARKET_CLOSE
    naive = pd.Timestamp.combine(pd.Timestamp(session).date(), clock)
    return naive.tz_localize(EASTERN).tz_convert("UTC")


def assert_no_lookahead(
    announced_utc: pd.Series, sessions: pd.Series, entry_timing: str
) -> None:
    """Fail loudly if any fill would have happened before its announcement.

    This is the guard that matters most in the repo. Every other bug produces
    a wrong number; this one produces a plausible-looking wrong number that
    survives review, because a strategy that trades on tomorrow's news looks
    exactly like a strategy that works.
    """
    if len(sessions) == 0:
        return
    fills = pd.DatetimeIndex([entry_timestamp(s, entry_timing) for s in sessions])
    announced = pd.DatetimeIndex(pd.to_datetime(announced_utc, utc=True))
    violations = fills <= announced
    if violations.any():
        first = int(np.argmax(violations))
        raise AssertionError(
            f"look-ahead: {int(violations.sum())} of {len(fills)} fills are at or before "
            f"their announcement; first is fill {fills[first]} vs announcement {announced[first]}"
        )


def build_event_panel(
    events: pd.DataFrame,
    panel: PricePanel,
    *,
    windows: tuple[int, ...] = config.RETURN_WINDOWS,
    benchmark: str = config.BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Attach reaction-window returns to each announcement.

    ``events`` needs ``ticker``, ``accepted_utc`` and whatever surprise
    columns the caller wants carried through. The result adds, for each
    horizon ``w`` in ``windows``:

    ``ret_d{w}``
        Raw return from the pre-announcement close to the close ``w`` sessions
        after the reaction session. ``w = 0`` is the announcement reaction
        itself, overnight gap included.
    ``abret_d{w}``
        The same, less the benchmark's return over the identical sessions.

    plus the decomposition of the reaction into the part that happens before
    anyone can trade and the part that does not:

    ``gap_ret``
        Pre-announcement close to the reaction session's open. Unfillable.
    ``open_to_close_ret``
        Reaction session's open to its close. The first tradeable slice.
    """
    if events.empty:
        return events.copy()

    dates = panel.dates
    n_sessions = len(dates)
    accepted = pd.to_datetime(events["accepted_utc"], utc=True)
    announced_et = accepted.dt.tz_convert(EASTERN).dt.tz_localize(None)
    timing = pd.Series(classify_timing(accepted), index=events.index)

    pos = reaction_positions(announced_et, timing, panel)

    frame = events.copy()
    frame["timing"] = timing.to_numpy()
    frame["announcement_date"] = announced_et.dt.normalize().to_numpy()
    frame["reaction_pos"] = pos

    # A reaction session must exist, must have a session before it to measure
    # the announcement move against, and must leave room for the longest
    # horizon requested.
    longest = max(windows)
    # One extra session of headroom so a mid-session release still has a full
    # window from its (later) opening-fill session.
    usable = (pos >= 1) & (pos + longest + 1 < n_sessions)
    frame = frame.loc[usable].copy()
    if frame.empty:
        return frame

    frame["reaction_date"] = dates[frame["reaction_pos"].to_numpy()]

    # When the market reacts and when a trader can fill are not the same
    # session for a mid-session release. The close of the reacting session is
    # still after the news, so a close fill is legitimate there; an opening
    # fill is not, because that auction happened before the release.
    frame["entry_pos_close"] = frame["reaction_pos"]
    frame["entry_pos_open"] = frame["reaction_pos"] + (frame["timing"] == INTRADAY).astype(int)
    frame = frame[frame["ticker"].isin(panel.close.columns)].copy()
    if frame.empty:
        return frame

    close = panel.close
    open_ = panel.open
    bench = close[benchmark].to_numpy() if benchmark in close.columns else None
    bench_open = open_[benchmark].to_numpy() if benchmark in open_.columns else None

    ticker_codes, ticker_index = pd.factorize(frame["ticker"])
    close_matrix = close[list(ticker_index)].to_numpy()
    open_matrix = open_[list(ticker_index)].to_numpy()

    rpos = frame["reaction_pos"].to_numpy()
    rows = np.arange(len(frame))

    pre_close = close_matrix[rpos - 1, ticker_codes]
    reaction_open = open_matrix[rpos, ticker_codes]
    reaction_close = close_matrix[rpos, ticker_codes]

    with np.errstate(invalid="ignore", divide="ignore"):
        frame["gap_ret"] = (reaction_open / pre_close - 1.0) * 100.0
        frame["open_to_close_ret"] = (reaction_close / reaction_open - 1.0) * 100.0

        # Benchmark-adjusted versions of the same two legs, so the whole
        # decomposition can be stated in one unit. The benchmark's own
        # overnight gap is subtracted from the stock's, which matters: on a
        # day the market gaps up a percent, an unadjusted gap of +1% is not a
        # reaction to anything.
        if bench_open is not None and bench is not None:
            frame["abgap_ret"] = frame["gap_ret"] - (bench_open[rpos] / bench[rpos - 1] - 1.0) * 100.0
            frame["abopen_to_close_ret"] = (
                frame["open_to_close_ret"] - (bench[rpos] / bench_open[rpos] - 1.0) * 100.0
            )
        else:  # pragma: no cover - benchmark is always present in practice
            frame["abgap_ret"] = np.nan
            frame["abopen_to_close_ret"] = np.nan

        for w in windows:
            tip = close_matrix[rpos + w, ticker_codes]
            raw = (tip / pre_close - 1.0) * 100.0
            frame[f"ret_d{w}"] = raw
            if bench is not None:
                bench_ret = (bench[rpos + w] / bench[rpos - 1] - 1.0) * 100.0
                frame[f"abret_d{w}"] = raw - bench_ret
            else:  # pragma: no cover - benchmark is always present in practice
                frame[f"abret_d{w}"] = np.nan

        # Drift measured from the reaction close, i.e. excluding the
        # announcement move entirely. This is what a next-close entry earns.
        for w in windows:
            if w == 0:
                continue
            tip = close_matrix[rpos + w, ticker_codes]
            drift = (tip / reaction_close - 1.0) * 100.0
            frame[f"drift_d{w}"] = drift
            if bench is not None:
                bench_ret = (bench[rpos + w] / bench[rpos] - 1.0) * 100.0
                frame[f"abdrift_d{w}"] = drift - bench_ret
            else:  # pragma: no cover
                frame[f"abdrift_d{w}"] = np.nan

    frame["season"] = (
        frame["announcement_date"].dt.year.astype(str)
        + "Q"
        + frame["announcement_date"].dt.quarter.astype(str)
    )
    del rows
    return frame.reset_index(drop=True)


def abnormal_return_matrix(
    events: pd.DataFrame,
    panel: PricePanel,
    *,
    max_days: int = 20,
    benchmark: str = config.BENCHMARK_TICKER,
    from_reaction_close: bool = True,
) -> np.ndarray:
    """Cumulative abnormal return for every event, day by day.

    Returns an ``(n_events, max_days + 1)`` array in percent. Column ``w`` is
    the cumulative abnormal return through ``w`` sessions after the reaction
    session.

    ``from_reaction_close`` measures from the close of the reaction session,
    excluding the announcement move — the path a next-close entry would ride.
    Setting it false measures from the pre-announcement close and so includes
    the jump.
    """
    close = panel.close
    codes, index = pd.factorize(events["ticker"])
    close_matrix = close[list(index)].to_numpy()
    bench = close[benchmark].to_numpy() if benchmark in close.columns else None

    rpos = events["reaction_pos"].to_numpy()
    base_pos = rpos if from_reaction_close else rpos - 1
    base = close_matrix[base_pos, codes]
    base_bench = bench[base_pos] if bench is not None else None

    out = np.full((len(events), max_days + 1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for w in range(max_days + 1):
            target = rpos + w
            valid = target < len(close)
            tip = np.where(valid, close_matrix[np.minimum(target, len(close) - 1), codes], np.nan)
            raw = (tip / base - 1.0) * 100.0
            if base_bench is not None:
                tip_bench = np.where(
                    valid, bench[np.minimum(target, len(close) - 1)], np.nan
                )
                raw = raw - (tip_bench / base_bench - 1.0) * 100.0
            out[:, w] = np.where(valid, raw, np.nan)
    return out
