"""Daily price panels.

Prices come from Yahoo via ``yfinance`` with ``auto_adjust=True``, so opens and
closes are both adjusted for splits and dividends on the same basis. Opens
matter here: an after-close announcement is first tradeable at the next open,
and the overnight gap is where a large part of the announcement reaction
happens. A close-only panel cannot separate the two.

The panel is cached as Parquet under ``data/cache``. It is regenerable and
large, so it is not committed; a small slice for tests and CI lives in
``data/sample``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

FIELDS = ("Open", "Close", "Volume")


@dataclass(frozen=True)
class PricePanel:
    """Aligned Open/Close/Volume frames sharing one trading calendar."""

    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def position_of(self, day: pd.Timestamp) -> int:
        """Index of the first session on or after ``day``.

        Returns ``len(dates)`` when ``day`` is past the end of the calendar,
        which callers treat as "no tradeable session".
        """
        return int(self.dates.searchsorted(pd.Timestamp(day).normalize(), side="left"))

    def subset(self, tickers: list[str]) -> PricePanel:
        keep = [t for t in tickers if t in self.close.columns]
        return PricePanel(self.open[keep], self.close[keep], self.volume[keep])


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    batch_size: int = 100,
    verbose: bool = True,
) -> PricePanel:
    """Download adjusted OHLCV for ``tickers`` in batches."""
    import yfinance as yf  # imported lazily so tests need not have it installed

    frames: dict[str, list[pd.DataFrame]] = {f: [] for f in FIELDS}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        if verbose:
            print(f"  prices {i + 1}-{i + len(batch)} of {len(tickers)}", flush=True)
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw is None or raw.empty:
            continue
        for field in FIELDS:
            if field not in raw.columns.get_level_values(0):
                continue
            frames[field].append(raw[field])

    if not frames["Close"]:
        raise RuntimeError("no price data returned; check network access and ticker list")

    panel = {f: pd.concat(frames[f], axis=1).sort_index() for f in FIELDS}
    close = panel["Close"]
    aligned = {f: panel[f].reindex(index=close.index, columns=close.columns) for f in FIELDS}
    out = PricePanel(open=aligned["Open"], close=close, volume=aligned["Volume"])
    return _drop_empty(out)


def _drop_empty(panel: PricePanel) -> PricePanel:
    """Drop tickers with no closing price anywhere in the window."""
    keep = panel.close.columns[panel.close.notna().any()].tolist()
    return panel.subset(keep)


def save_panel(panel: PricePanel, directory: Path, *, source_as_of: str | None = None) -> None:
    """Write the panel to Parquet with a provenance sidecar."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in (("open", panel.open), ("close", panel.close), ("volume", panel.volume)):
        frame.to_parquet(directory / f"{name}.parquet")
    config.write_provenance(
        directory / "close.parquet",
        command="python main.py fetch",
        data_source="Yahoo Finance via yfinance, auto_adjust=True",
        as_of=source_as_of or date.today().isoformat(),
        extra={
            "n_tickers": len(panel.tickers),
            "first_session": str(panel.dates.min().date()),
            "last_session": str(panel.dates.max().date()),
        },
    )


def load_panel(directory: Path) -> PricePanel:
    """Read a panel previously written by :func:`save_panel`."""
    frames = {}
    for name in ("open", "close", "volume"):
        path = directory / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run `python main.py fetch` first")
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frames[name] = frame.sort_index()
    return PricePanel(open=frames["open"], close=frames["close"], volume=frames["volume"])


def forward_return(prices: np.ndarray, start_pos: int, end_pos: int) -> float:
    """Simple return between two positions of a price array, in percent."""
    if start_pos < 0 or end_pos >= len(prices):
        return float("nan")
    base = prices[start_pos]
    tip = prices[end_pos]
    if not np.isfinite(base) or not np.isfinite(tip) or base == 0:
        return float("nan")
    return (tip / base - 1.0) * 100.0
