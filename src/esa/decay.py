"""Has the drift decayed?

Confirming that post-earnings announcement drift exists is not an interesting
result in 2026; Ball & Brown documented it in 1968 and it has been in every
anomalies survey since. The live question is whether it still pays, and the
way to look at that is a rolling window rather than a single full-sample
average that mixes 2012 with 2025.

The window is three years wide and steps one earnings season at a time. Three
years is twelve seasons, which is enough for the cluster bootstrap inside each
window to say something and short enough that a regime change is visible
rather than averaged away.

Windows overlap heavily, so the confidence bands of neighbouring points are
not independent and the curve should be read for its level and slope, not for
whether any single point clears zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import SeasonStats

COLUMNS = ["season", "end_date", "spread", "ci_low", "ci_high", "n_events", "n_beat", "n_miss"]


def rolling_spread(
    events: pd.DataFrame,
    metric: str,
    *,
    window_seasons: int = 12,
    cluster: str = "season",
    n_boot: int = 2_000,
    seed: int = 0,
    min_events: int = 50,
) -> pd.DataFrame:
    """Beat-minus-Miss spread over a rolling window, with a bootstrap band.

    Each window is resampled at the season level exactly as the full-sample
    test is, so the band widens where the sample thins rather than pretending
    to a precision the window does not have.

    ``min_events`` skips a window too thin to say anything. It is set low
    enough that it never binds on the full sample — the thinnest real window
    holds several hundred events — and exists for reduced runs.

    Returns one row per window: the last season it covers, the point estimate,
    the 5th and 95th bootstrap percentiles and the event count.
    """
    stats_ = SeasonStats.build(events, metric, cluster)
    seasons = stats_.keys
    if len(seasons) < window_seasons:
        return pd.DataFrame(columns=COLUMNS)

    rng = np.random.default_rng(seed)
    rows = []
    for stop in range(window_seasons, len(seasons) + 1):
        window = np.arange(stop - window_seasons, stop)
        n_beat = float(stats_.beat_n[window].sum())
        n_miss = float(stats_.miss_n[window].sum())
        if n_beat + n_miss < min_events:
            continue
        point = stats_.spread(window)
        picks = window[rng.integers(0, window_seasons, size=(n_boot, window_seasons))]
        draws = stats_.spread_many(picks)
        draws = draws[np.isfinite(draws)]
        lo, hi = np.percentile(draws, [5, 95]) if len(draws) else (np.nan, np.nan)
        label = str(seasons[stop - 1])
        rows.append(
            {
                "season": label,
                "end_date": _season_end(label),
                "spread": point,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_events": int(n_beat + n_miss),
                "n_beat": int(n_beat),
                "n_miss": int(n_miss),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _season_end(label: str) -> pd.Timestamp:
    """Last calendar day of a ``YYYYQn`` label."""
    year, quarter = label.split("Q")
    return pd.Period(year=int(year), quarter=int(quarter), freq="Q").end_time.normalize()


def trend_test(curve: pd.DataFrame) -> dict[str, float]:
    """Slope of the rolling spread against time, as a descriptive statistic.

    The windows overlap, so the ordinary standard error of this slope is
    meaningless and is deliberately not reported. The slope itself still says
    how many percentage points a decade of calendar time is associated with,
    which is the number the decay figure is really about.
    """
    if "spread" not in curve.columns:
        return {"slope_pp_per_decade": float("nan"), "first": float("nan"), "last": float("nan")}
    usable = curve.dropna(subset=["spread"]).copy()
    if len(usable) < 3:
        return {"slope_pp_per_decade": float("nan"), "first": float("nan"), "last": float("nan")}
    # Accept a curve read back from CSV, where the dates arrive as strings.
    usable["end_date"] = pd.to_datetime(usable["end_date"])
    years = (usable["end_date"] - usable["end_date"].iloc[0]).dt.days.to_numpy() / 365.25
    slope = float(np.polyfit(years, usable["spread"].to_numpy(), 1)[0])
    return {
        "slope_pp_per_decade": slope * 10.0,
        "first": float(usable["spread"].iloc[0]),
        "last": float(usable["spread"].iloc[-1]),
    }
