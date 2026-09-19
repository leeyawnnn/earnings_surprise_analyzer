"""Command line entry point.

    python main.py fetch      # download filings, prices and factors into data/cache
    python main.py study      # build the event panel and write docs/results
    python main.py figures    # redraw every figure from the last study
    python main.py all        # study then figures

``snapshot-universe`` refreshes the committed index membership file. It is
deliberately separate: refreshing it changes the universe and therefore every
published number, so it should never happen as a side effect of an analysis
run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from esa import config, pipeline, study, universe
from esa.config import StudyConfig
from esa.plotting import figures


def _print_headline(results: study.StudyResults) -> None:
    events = results.events
    print()
    print(f"  events            {len(events):,} announcements, {events['ticker'].nunique()} companies")
    print(
        f"  period            {events['announcement_date'].min():%Y-%m-%d} to "
        f"{events['announcement_date'].max():%Y-%m-%d}"
    )
    print(f"  seasons           {events['season'].nunique()}")
    print()
    headline = results.tests[results.tests["metric"] == "abdrift_d20"]
    for _, row in headline.iterrows():
        print(
            f"  {row['method'][:46]:<46} spread {row['estimate']:+.3f} pp   p = {row['p_value']:.4f}"
        )
    print()
    for _, row in results.backtest_summary.iterrows():
        print(
            f"  backtest {row['entry_timing']:<11} net {row['total_return_pct']:+8.2f}%   "
            f"Sharpe {row['sharpe']:+.2f}   maxDD {row['max_drawdown_pct']:.1f}%   "
            f"SPY {row['benchmark_total_return_pct']:+.1f}%"
        )
    for timing, bps in results.breakeven_bps.items():
        print(f"  break-even cost   {timing:<11} {bps:.1f} bp round trip")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__)
    parser.add_argument(
        "command",
        choices=["fetch", "study", "figures", "all", "snapshot-universe"],
        help="which stage to run",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="run against the committed sample in data/sample instead of the full cache",
    )
    args = parser.parse_args(argv)

    verbose = not args.quiet
    cfg = StudyConfig()
    config.ensure_dirs()

    if args.command == "snapshot-universe":
        frame = universe.snapshot_sp500(config.UNIVERSE_SNAPSHOT)
        print(f"wrote {config.UNIVERSE_SNAPSHOT} with {len(frame)} constituents")
        return 0

    if args.command == "fetch":
        pipeline.fetch_all(cfg, verbose=verbose)
        return 0

    # The sample run exercises the whole pipeline on the committed slice and
    # writes nothing into the published directories, so CI cannot silently
    # overwrite a result table with one computed from 20 companies.
    cache_dir = config.SAMPLE_DIR if args.sample else None
    results_dir = config.OUTPUT_DIR / "sample_results" if args.sample else None
    figures_dir = config.OUTPUT_DIR / "sample_figures" if args.sample else None
    if args.sample:
        cfg = cfg.with_(n_bootstrap=400, event_start_date="2013-01-01")

    if args.command in ("study", "all", "figures"):
        results = study.run_study(
            cfg, verbose=verbose, cache_dir=cache_dir, results_dir=results_dir
        )
        if verbose and args.command != "figures":
            _print_headline(results)
        if args.command in ("all", "figures"):
            panel = pipeline.load_raw(cache_dir).panel
            written = figures.build_all(results, panel, figures_dir)
            if verbose:
                for path in written:
                    print(f"  {Path(path).name}")
        return 0

    return 1  # pragma: no cover - argparse rejects anything else


if __name__ == "__main__":
    sys.exit(main())
