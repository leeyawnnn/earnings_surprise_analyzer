"""Measuring how surprising a quarterly result was.

Two definitions live here, and the difference between them is the single most
important methodological choice in this repo.

**Standardised unexpected earnings (SUE).** Forecast this quarter's EPS from
the same quarter a year ago plus the firm's recent drift, then scale the error
by how variable that error has been for this firm. Bernard & Thomas (1989)
use this form, and it has one property an analyst-based measure cannot match
here: every input is a number the company itself published, on a date EDGAR
records, so the forecast can be rebuilt exactly as it stood the day before the
announcement.

**Percent deviation from consensus.** The measure the first version of this
repo used, kept so the two can be compared. It is closer to what moves a stock
— the market trades against expectations, not against last year — but a free
consensus feed reports today's consensus, not the consensus that stood before
the announcement. If estimates were revised afterwards, the "surprise" is
partly built from information that did not exist at the time.

Neither measure is the truth. SUE is point-in-time and noisy; consensus
deviation is economically sharper and contaminated. The study reports SUE and
uses consensus deviation only as a cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

#: A seasonal lag must land roughly one year back. Fiscal calendars drift by a
#: few days and 52/53-week retail years by up to a week, but a gap outside
#: this range means a quarter is missing from the series.
SEASONAL_LAG_DAYS = (330, 400)


def categorize(value: float, *, beat: float, miss: float) -> str:
    """Bucket a surprise measure.

    Boundaries are exclusive on both sides: a value exactly at the threshold
    is In-Line. That keeps the three buckets disjoint and makes the common
    "surprise of exactly zero" case fall where it belongs.
    """
    if pd.isna(value):
        return "Unknown"
    if value > beat:
        return "Beat"
    if value < miss:
        return "Miss"
    return "In-Line"


def categorize_series(values: pd.Series, *, beat: float, miss: float) -> pd.Series:
    """Vectorised :func:`categorize`."""
    out = pd.Series("In-Line", index=values.index, dtype="object")
    out[values > beat] = "Beat"
    out[values < miss] = "Miss"
    out[values.isna()] = "Unknown"
    return out


def percent_surprise(actual: float, estimate: float, *, cap: float = 200.0) -> float:
    """Percent deviation from an estimate, scaled by ``|estimate|``.

    Dividing by the absolute estimate keeps the sign meaningful when the
    estimate is negative: a loss that came in smaller than feared is a
    positive surprise. A near-zero estimate makes the ratio explode, so tiny
    denominators return NaN rather than a number that would dominate every
    cross-sectional statistic it enters, and the result is capped.
    """
    if pd.isna(actual) or pd.isna(estimate):
        return float("nan")
    if abs(estimate) < 0.01:
        return float("nan")
    value = (actual - estimate) / abs(estimate) * 100.0
    return float(np.clip(value, -cap, cap))


def standardized_unexpected_earnings(
    eps: pd.DataFrame,
    *,
    history_quarters: int = config.SUE_HISTORY_QUARTERS,
    include_drift: bool = True,
) -> pd.DataFrame:
    """Compute SUE per firm-quarter from a panel of reported EPS.

    ``eps`` needs ``ticker``, ``end`` (fiscal period end), ``val`` (EPS) and
    ``filed``. The result adds:

    ``seasonal_diff``
        EPS this quarter less EPS four quarters ago.
    ``drift``
        Mean seasonal difference over the previous ``history_quarters``
        quarters — the firm's recent earnings growth, which the naive seasonal
        random walk would misread as a surprise every quarter.
    ``sue_sigma``
        Standard deviation of those same differences.
    ``sue``
        ``(seasonal_diff - drift) / sue_sigma``, or ``seasonal_diff /
        sue_sigma`` when ``include_drift`` is false.
    ``history_known_by``
        The latest filing date among every input to this quarter's forecast.
        The event study drops any announcement that does not strictly postdate
        it, which is what keeps the measure point-in-time.

    Rows without a full history, or with a degenerate scale, come back as NaN
    rather than being dropped, so that the caller can count what was lost.
    """
    required = {"ticker", "end", "val", "filed"}
    missing = required - set(eps.columns)
    if missing:
        raise ValueError(f"eps panel is missing columns: {sorted(missing)}")

    out: list[pd.DataFrame] = []
    for ticker, group in eps.sort_values(["ticker", "end"]).groupby("ticker", sort=False):
        frame = group.reset_index(drop=True).copy()

        lag4 = frame["val"].shift(4)
        lag4_end = frame["end"].shift(4)
        lag4_filed = frame["filed"].shift(4)
        gap_days = (frame["end"] - lag4_end).dt.days
        valid_lag = gap_days.between(*SEASONAL_LAG_DAYS)

        seasonal = (frame["val"] - lag4).where(valid_lag)
        frame["seasonal_diff"] = seasonal

        # The forecast may only use differences already public, so the window
        # is shifted by one and closed on the left.
        prior = seasonal.shift(1)
        rolling = prior.rolling(history_quarters, min_periods=history_quarters)
        frame["drift"] = rolling.mean()
        frame["sue_sigma"] = rolling.std(ddof=1)

        numerator = seasonal - frame["drift"] if include_drift else seasonal
        sigma = frame["sue_sigma"].where(frame["sue_sigma"] > 1e-9)
        frame["sue"] = numerator / sigma

        # The moment the forecast becomes computable: the latest filing date
        # among the seasonal lag and the history window.
        #
        # The running maximum is expanding rather than windowed. Filing dates
        # are non-decreasing in practice, so the two agree; where they do not,
        # the expanding version returns the later date and therefore the
        # stricter guard, which is the right way to be wrong here.
        frame["history_known_by"] = pd.concat(
            [lag4_filed, frame["filed"].shift(1).cummax()], axis=1
        ).max(axis=1)

        frame["ticker"] = ticker
        out.append(frame)

    if not out:
        return eps.assign(sue=np.nan)
    return pd.concat(out, ignore_index=True)


def bucket_summary(events: pd.DataFrame, column: str = "category") -> pd.DataFrame:
    """Counts and shares per surprise bucket, for the README's sample table."""
    counts = events[column].value_counts()
    return pd.DataFrame(
        {
            "bucket": counts.index,
            "n": counts.to_numpy(),
            "share_pct": (counts.to_numpy() / len(events) * 100.0).round(2),
        }
    ).sort_values("bucket").reset_index(drop=True)
