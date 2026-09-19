"""Download Yahoo's consensus estimates for the universe.

Separate from ``python main.py fetch`` on purpose. The consensus figures are
not point-in-time and are used only for the comparison in
``docs/results/surprise_definition_comparison.csv``; nothing in the headline
result depends on them. See :mod:`esa.consensus` for what the field is and why
it cannot carry the study.

    python scripts/fetch_consensus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esa import config
from esa.consensus import fetch_consensus, save_consensus
from esa.universe import load_universe


def main() -> int:
    config.ensure_dirs()
    tickers = load_universe()["ticker"].tolist()
    frame = fetch_consensus(tickers)
    if frame.empty:
        print("no consensus data returned", file=sys.stderr)
        return 1
    path = save_consensus(frame)
    print(f"  rows            {len(frame):,}")
    print(f"  companies       {frame['ticker'].nunique()}")
    print(f"  earliest        {frame['announced_date'].min():%Y-%m-%d}")
    print(f"  latest          {frame['announced_date'].max():%Y-%m-%d}")
    print(f"  written         {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
