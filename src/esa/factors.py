"""Daily Fama-French factors and the risk-free rate.

The calendar-time portfolio test regresses a daily strategy return on factor
returns, so it needs a factor set that is not derived from the same prices the
strategy trades. Ken French's data library publishes daily Mkt-RF, SMB, HML
and the risk-free rate, plus a separate momentum file.

Momentum matters for this particular strategy. A long-beat/short-miss book
built on recent earnings news is mechanically tilted toward recent winners, so
an alpha measured against the market alone would partly be a momentum
loading in disguise.

The risk-free series here is the one-month Treasury bill return already
expressed as a daily simple rate, which is what the Sharpe calculation needs.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FACTORS_FILE = "F-F_Research_Data_Factors_daily_CSV.zip"
MOMENTUM_FILE = "F-F_Momentum_Factor_daily_CSV.zip"

FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "mom"]


def _read_french_zip(content: bytes) -> pd.DataFrame:
    """Parse a Ken French daily CSV out of its zip wrapper.

    The files open with a prose header of varying length and close with a
    copyright block, so rows are selected by shape — an eight-digit date
    followed by numeric columns — rather than by a fixed skiprows count that
    would silently break when the header is reworded.
    """
    archive = zipfile.ZipFile(io.BytesIO(content))
    text = archive.read(archive.namelist()[0]).decode("latin-1")

    rows: list[list[str]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or len(parts[0]) != 8 or not parts[0].isdigit():
            continue
        rows.append(parts)
    if not rows:
        raise ValueError("no data rows found in Ken French archive")

    width = max(len(r) for r in rows)
    frame = pd.DataFrame([r + [""] * (width - len(r)) for r in rows])
    frame = frame.set_index(0)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "date"
    return frame.apply(pd.to_numeric, errors="coerce")


def download_factors(cache_dir: Path | None = None, *, timeout: int = 90) -> pd.DataFrame:
    """Fetch and merge the three-factor and momentum files.

    Values are returned in decimal (the source publishes percent), indexed by
    trading date, with columns ``mkt_rf, smb, hml, mom, rf``.
    """
    cache_dir = cache_dir or config.CACHE_DIR / "factors"
    cache_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": config.SEC_USER_AGENT}
    three = requests.get(FRENCH_BASE + FACTORS_FILE, headers=headers, timeout=timeout)
    three.raise_for_status()
    mom = requests.get(FRENCH_BASE + MOMENTUM_FILE, headers=headers, timeout=timeout)
    mom.raise_for_status()

    ff = _read_french_zip(three.content).iloc[:, :4]
    ff.columns = ["mkt_rf", "smb", "hml", "rf"]
    mm = _read_french_zip(mom.content).iloc[:, :1]
    mm.columns = ["mom"]

    merged = ff.join(mm, how="left") / 100.0
    merged = merged[["mkt_rf", "smb", "hml", "mom", "rf"]]
    path = cache_dir / "ff_daily.parquet"
    merged.to_parquet(path)
    config.write_provenance(
        path,
        command="python main.py fetch",
        data_source=f"Kenneth R. French Data Library: {FACTORS_FILE}, {MOMENTUM_FILE}",
        as_of=str(merged.index.max().date()),
        extra={"first_date": str(merged.index.min().date()), "n_days": int(len(merged))},
    )
    return merged


def load_factors(cache_dir: Path | None = None) -> pd.DataFrame:
    """Read the cached factor panel."""
    cache_dir = cache_dir or config.CACHE_DIR / "factors"
    path = cache_dir / "ff_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `python main.py fetch` first")
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    return frame
