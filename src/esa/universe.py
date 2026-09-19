"""Index membership.

The study needs to know which companies to look at and, ideally, which ones
were in the index on any given historical date. A free point-in-time
constituent list does not exist, so this module does what can honestly be done
with a current snapshot: it keeps the date each name *entered* the index and
refuses to use a name before that date.

That removes backfill bias — the error of studying a 2013 earnings report from
a company that only joined the index in 2021 — but it cannot remove
survivorship bias, because companies that left the index are simply absent
from the snapshot. See :func:`members_on` for the precise guarantee.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from . import config

#: Source of the membership snapshot. The table is maintained in public and is
#: the only free list that carries an "added on" column.
SP500_TABLE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_COLUMNS = ["ticker", "name", "sector", "cik", "date_added"]


def _yahoo_symbol(symbol: str) -> str:
    """Convert an index-table symbol to the form price vendors use.

    Class shares are written ``BRK.B`` in the index table and ``BRK-B`` by
    Yahoo; sending the dotted form returns an empty series rather than an
    error, which is the kind of silent gap that quietly shrinks a sample.
    """
    return symbol.strip().upper().replace(".", "-")


def snapshot_sp500(dest: Path, *, timeout: int = 60) -> pd.DataFrame:
    """Download the current constituent table and write it to ``dest``.

    Run deliberately, not as part of the analysis: the committed snapshot is
    what makes a run reproducible, and refreshing it changes the universe.
    """
    response = requests.get(
        SP500_TABLE_URL,
        headers={"User-Agent": config.SEC_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    table = pd.read_html(io.StringIO(response.text), match="Symbol")[0]

    frame = pd.DataFrame(
        {
            "ticker": [_yahoo_symbol(s) for s in table["Symbol"]],
            "name": table["Security"].astype(str).str.strip(),
            "sector": table["GICS Sector"].astype(str).str.strip(),
            "cik": table["CIK"].astype(int),
            "date_added": pd.to_datetime(table["Date added"], errors="coerce"),
        }
    ).sort_values("ticker")

    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    config.write_provenance(
        dest,
        command="python main.py snapshot-universe",
        data_source=SP500_TABLE_URL,
        as_of=date.today().isoformat(),
        extra={
            "n_constituents": len(frame),
            "n_missing_date_added": int(frame["date_added"].isna().sum()),
        },
    )
    return frame


def load_universe(path: Path | None = None) -> pd.DataFrame:
    """Read the committed membership snapshot."""
    path = path or config.UNIVERSE_SNAPSHOT
    frame = pd.read_csv(path, parse_dates=["date_added"])
    missing = set(_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def members_on(universe: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    """Tickers that had joined the index on or before ``as_of``.

    The guarantee is one-sided. A name is excluded before its documented join
    date, so the sample contains no company that was not yet in the index. A
    name that was *removed* from the index after ``as_of`` is missing from the
    snapshot entirely and cannot be recovered here, so the sample still
    over-represents survivors.

    Rows with no recorded join date are treated as always-members, which is
    the conservative reading for the handful of pre-1970 additions whose dates
    the source leaves blank.
    """
    added = universe["date_added"]
    eligible = added.isna() | (added <= as_of)
    return universe.loc[eligible, "ticker"].tolist()


def eligibility_mask(universe: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Boolean mask over ``events`` keeping only post-join announcements.

    ``events`` needs ``ticker`` and ``announcement_date`` columns.
    """
    joined = universe.set_index("ticker")["date_added"]
    event_join = events["ticker"].map(joined)
    return event_join.isna() | (event_join <= events["announcement_date"])
