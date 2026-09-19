"""Cut the committed CI sample out of the full download cache.

The full panel is five hundred companies of prices and filings and does not
belong in a repository. But a continuous integration run that never touches
real data only proves the code imports, so a small slice is committed and the
whole pipeline runs against it on every push.

The slice is deliberately not a random one. It keeps companies that between
them exercise the awkward paths: a non-calendar fiscal year, a derived fourth
quarter, releases before the open and after the close, and at least one
mid-session release, which is the case where the reacting session and the
first fillable session differ.

Regenerate with:

    python scripts/build_sample.py

after ``python main.py fetch``. It is a maintenance command, not part of the
analysis: the committed sample changes only when someone decides it should.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esa import config
from esa.events import classify_timing
from esa.prices import load_panel, save_panel
from esa.universe import load_universe

#: Twenty-four companies across eight sectors. AAPL, MSFT and WMT are here for
#: their September and January fiscal year ends; INTC for quarters where
#: earnings per share sit near zero.
SAMPLE_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "INTC",
    "CSCO",
    "JPM",
    "BAC",
    "GS",
    "JNJ",
    "PFE",
    "UNH",
    "XOM",
    "CVX",
    "PG",
    "KO",
    "WMT",
    "COST",
    "HD",
    "MCD",
    "NKE",
    "CAT",
    "UPS",
    "T",
    "VZ",
]


def main() -> int:
    cache = config.CACHE_DIR
    sample = config.SAMPLE_DIR
    if not (cache / "sec" / "announcements.parquet").exists():
        print("no download cache found; run `python main.py fetch` first", file=sys.stderr)
        return 1

    sample.mkdir(parents=True, exist_ok=True)
    (sample / "sec").mkdir(exist_ok=True)
    (sample / "factors").mkdir(exist_ok=True)

    members = load_universe()
    keep = members[members["ticker"].isin(SAMPLE_TICKERS)]
    keep.to_csv(sample / "universe.csv", index=False)

    for name in ("announcements", "quarterly_eps"):
        frame = pd.read_parquet(cache / "sec" / f"{name}.parquet")
        frame[frame["ticker"].isin(SAMPLE_TICKERS)].reset_index(drop=True).to_parquet(
            sample / "sec" / f"{name}.parquet"
        )

    panel = load_panel(cache / "prices")
    wanted = [*SAMPLE_TICKERS, config.BENCHMARK_TICKER]
    save_panel(panel.subset(wanted), sample / "prices", source_as_of=config.StudyConfig().end_date)

    # The factor file runs back to 1926. The study window is all CI needs.
    factors = pd.read_parquet(cache / "factors" / "ff_daily.parquet")
    factors = factors[factors.index >= pd.Timestamp(config.StudyConfig().start_date)]
    factors.to_parquet(sample / "factors" / "ff_daily.parquet", compression="zstd")
    for meta in (cache / "factors").glob("*.meta.json"):
        shutil.copy(meta, sample / "factors" / meta.name)

    announced = pd.read_parquet(sample / "sec" / "announcements.parquet")
    timing = classify_timing(announced["accepted_utc"])
    total = sum(p.stat().st_size for p in sample.rglob("*") if p.is_file())
    print(f"  companies       {keep['ticker'].nunique()}")
    print(f"  announcements   {len(announced):,}")
    print(f"  timing mix      {dict(pd.Series(timing).value_counts())}")
    print(f"  sample size     {total / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
