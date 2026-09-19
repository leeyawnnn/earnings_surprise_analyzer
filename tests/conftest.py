"""Synthetic fixtures.

Everything here is built by hand so that the expected answer can be worked out
on paper. No test in this suite touches the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esa.prices import PricePanel

#: A short, gap-free calendar of business days. Using real weekday dates means
#: the weekend-handling paths get exercised without inventing a calendar.
SESSIONS = pd.bdate_range("2020-01-02", periods=60)


def make_panel(
    closes: dict[str, list[float]] | None = None,
    opens: dict[str, list[float]] | None = None,
    sessions: pd.DatetimeIndex | None = None,
) -> PricePanel:
    """Build a :class:`PricePanel` from explicit price paths.

    Defaults give ``AAA`` a flat 100 and ``SPY`` a flat 100, so any deviation a
    test introduces is the whole of the measured return.
    """
    sessions = sessions if sessions is not None else SESSIONS
    n = len(sessions)
    closes = closes or {"AAA": [100.0] * n, "SPY": [100.0] * n}
    opens = opens or {t: list(v) for t, v in closes.items()}
    close = pd.DataFrame(closes, index=sessions)
    open_ = pd.DataFrame(opens, index=sessions).reindex(columns=close.columns)
    return PricePanel(open=open_, close=close)


def utc(stamp: str) -> pd.Timestamp:
    """Eastern wall-clock string to a UTC timestamp, as EDGAR would record it."""
    return pd.Timestamp(stamp, tz="America/New_York").tz_convert("UTC")


@pytest.fixture
def flat_panel() -> PricePanel:
    return make_panel()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
