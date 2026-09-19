"""The study: every published table comes from here.

One entry point, :func:`run_study`, produces the result set the README quotes.
Each table is written to ``docs/results`` with a provenance sidecar, so a
number in the prose can always be traced back to a file, a command and a
commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, decay, inference, pipeline
from .backtest import BacktestResult, cost_sweep, run_backtest
from .config import StudyConfig
from .prices import PricePanel

#: Horizons reported in the headline table. Day 0 is the announcement reaction
#: and is not drift; it is included so the reader can see how much of the
#: total move happens before any drift can be traded.
HEADLINE_METRICS = ("abret_d0", "abret_d5", "abret_d20")
#: The same horizons measured from the reaction close, which is what a
#: next-close entry actually earns. This is the strict PEAD measure.
DRIFT_METRICS = ("abdrift_d1", "abdrift_d5", "abdrift_d10", "abdrift_d20")


@dataclass
class StudyResults:
    """Every table the README and the figures read."""

    events: pd.DataFrame
    audit: dict[str, int]
    tests: pd.DataFrame
    calendar_time: pd.DataFrame
    calendar_alpha: pd.DataFrame
    backtests: dict[str, BacktestResult] = field(default_factory=dict)
    backtest_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    breakeven_bps: dict[str, float] = field(default_factory=dict)
    decay_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    decay_trend: dict[str, float] = field(default_factory=dict)
    power: pd.DataFrame = field(default_factory=pd.DataFrame)
    timing: pd.DataFrame = field(default_factory=pd.DataFrame)
    decomposition: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_inference(events: pd.DataFrame, cfg: StudyConfig, *, verbose: bool = True) -> pd.DataFrame:
    """Naive and corrected tests for every reported horizon, side by side."""
    rows: list[inference.TestResult] = []
    metrics = [*HEADLINE_METRICS, *DRIFT_METRICS]
    for metric in metrics:
        if metric not in events.columns:
            continue
        if verbose:
            print(f"  inference: {metric}", flush=True)
        rows.append(inference.welch_spread(events, metric))
        rows.append(
            inference.block_bootstrap_spread(
                events, metric, n_boot=cfg.n_bootstrap, seed=cfg.random_seed
            )
        )
        rows.append(
            inference.wild_cluster_bootstrap(
                events, metric, n_boot=cfg.n_bootstrap, seed=cfg.random_seed
            )
        )
    frame = inference.results_frame(rows)
    frame["icc_season"] = frame["metric"].map(
        {m: inference.intra_cluster_correlation(events, m) for m in frame["metric"].unique()}
    )
    return frame


def run_calendar_time(
    events: pd.DataFrame, panel: PricePanel, factor_panel: pd.DataFrame, cfg: StudyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily calendar-time portfolio returns and the factor-model alpha."""
    portfolio = inference.calendar_time_returns(events, panel, window=cfg.hold_days)
    rows = []
    for column, excess in (("spread_ret", False), ("long_ret", True), ("short_ret", True)):
        result, loadings = inference.calendar_time_alpha(
            portfolio, factor_panel, column=column, excess=excess, hac_lags=cfg.hold_days
        )
        record = result.as_dict()
        record["alpha_annualised_pct"] = result.estimate * 252.0 * 100.0
        record.update({f"beta_{k}": float(v) for k, v in loadings.items()})
        rows.append(record)
    return portfolio, pd.DataFrame(rows)


def run_backtests(
    events: pd.DataFrame, panel: PricePanel, factor_panel: pd.DataFrame, cfg: StudyConfig
) -> tuple[dict[str, BacktestResult], pd.DataFrame]:
    """Run the book under each entry-timing convention."""
    risk_free = factor_panel["rf"] if "rf" in factor_panel.columns else None
    results: dict[str, BacktestResult] = {}
    rows = []
    for timing in ("next_open", "next_close"):
        result = run_backtest(events, panel, cfg, entry_timing=timing)
        if risk_free is not None and not result.daily.empty:
            from .backtest import compute_metrics
            from .costs import CostModel

            traded = result.daily[result.daily["net_ret"].notna()]
            result.metrics = compute_metrics(
                traded, result.trades, panel, CostModel(cfg.costs), risk_free=risk_free
            )
        results[timing] = result
        rows.append({"entry_timing": timing, **result.metrics})
    return results, pd.DataFrame(rows)


def decompose_reaction(events: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    """Split the total move into the part nobody can trade and the parts they can.

    ``gap`` is the pre-announcement close to the reaction open: it is over
    before the market opens and is not available to anyone reacting to the
    release. ``open_to_close`` is the first tradeable slice. ``drift`` is
    everything after that first close, which is the part PEAD is about.

    Every column is benchmark-adjusted. Mixing a raw drift with an abnormal
    spread would make the per-bucket rows look like a result when they are
    mostly the market: over twenty sessions of this sample the index itself
    returns well over a percent, which swamps the drift being measured.
    """
    rows = []
    for bucket in ("Beat", "In-Line", "Miss"):
        subset = events[events["category"] == bucket]
        if subset.empty:
            continue
        rows.append(
            {
                "category": bucket,
                "n": len(subset),
                "gap_pct": float(subset["abgap_ret"].mean()),
                "open_to_close_pct": float(subset["abopen_to_close_ret"].mean()),
                f"drift_d{cfg.hold_days}_pct": float(subset[f"abdrift_d{cfg.hold_days}"].mean()),
                f"total_d{cfg.hold_days}_pct": float(subset[f"abret_d{cfg.hold_days}"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    if {"Beat", "Miss"}.issubset(set(frame.get("category", []))):
        beat = frame[frame["category"] == "Beat"].iloc[0]
        miss = frame[frame["category"] == "Miss"].iloc[0]
        spread = {"category": "Beat - Miss", "n": int(beat["n"] + miss["n"])}
        for column in frame.columns:
            if column in ("category", "n"):
                continue
            spread[column] = float(beat[column] - miss[column])
        frame = pd.concat([frame, pd.DataFrame([spread])], ignore_index=True)
    return frame


def run_power(events: pd.DataFrame, tests: pd.DataFrame, metric: str) -> pd.DataFrame:
    """How much sample the observed effect would need to be reliably detectable.

    Two answers, and the gap between them is the point. The naive one treats
    every announcement as an independent observation. The corrected one
    inflates the variance by the ratio the season bootstrap actually measured,
    which is internally consistent with the p-values reported alongside it.

    The intra-cluster correlation is reported too, but it is not what drives
    the correction here. It describes dependence in the *level* of abnormal
    returns, and a Beat-minus-Miss spread differences most of that out, so the
    design effect it implies is far larger than the loss of precision the
    bootstrap finds. Using it would overstate the correction.
    """
    usable = events.dropna(subset=[metric])
    if usable.empty:
        return pd.DataFrame()
    naive = tests[(tests["metric"] == metric) & (tests["method"].str.startswith("welch"))]
    clustered = tests[(tests["metric"] == metric) & (tests["method"].str.startswith("block"))]
    if naive.empty or clustered.empty:
        return pd.DataFrame()

    effect = float(naive.iloc[0]["estimate"])
    se_naive = float(naive.iloc[0]["std_error"])
    se_cluster = float(clustered.iloc[0]["std_error"])
    inflation = (se_cluster / se_naive) ** 2 if se_naive > 0 else float("nan")

    tradeable = usable[usable["category"].isin(["Beat", "Miss"])]
    pooled_sd = float(tradeable[metric].std(ddof=1))
    icc = inference.intra_cluster_correlation(tradeable, metric)
    sizes = tradeable.groupby("season").size().to_numpy()
    mean_size = float(np.mean(sizes)) if len(sizes) else float("nan")

    return pd.DataFrame(
        [
            {
                "metric": metric,
                "observed_spread_pct": effect,
                "pooled_sd_pct": pooled_sd,
                "n_events_observed": int(len(tradeable)),
                "n_seasons": int(len(sizes)),
                "mean_events_per_season": mean_size,
                "se_naive": se_naive,
                "se_clustered": se_cluster,
                "variance_inflation": inflation,
                "icc_season": icc,
                "design_effect_from_icc": 1.0 + (mean_size - 1.0) * icc,
                "effective_sample_size": len(tradeable) / inflation if inflation > 0 else float("nan"),
                "n_for_80pct_power_independent": inference.required_sample_size(
                    effect, pooled_sd, variance_inflation=1.0
                ),
                "n_for_80pct_power_clustered": inference.required_sample_size(
                    effect, pooled_sd, variance_inflation=inflation
                ),
                "achieved_power_clustered": inference.achieved_power(effect, se_cluster),
                # What the first version of this repo was working with: four
                # quarters of a 25-name universe, about a hundred events.
                "achieved_power_at_100_events": inference.achieved_power(
                    effect, pooled_sd * float(np.sqrt(4.0 / 100.0 * inflation))
                ),
            }
        ]
    )


def run_study(
    cfg: StudyConfig,
    *,
    verbose: bool = True,
    cache_dir: Path | None = None,
    results_dir: Path | None = None,
    output_dir: Path | None = None,
) -> StudyResults:
    """Build the event panel, run everything, and write the result tables."""
    raw = pipeline.load_raw(cache_dir)
    if verbose:
        print("Building event panel")
    events, audit = pipeline.build_events(raw, cfg, verbose=verbose)
    pipeline.save_events(events, cfg, output_dir)

    if verbose:
        print("Running inference")
    tests = run_inference(events, cfg, verbose=verbose)

    if verbose:
        print("Calendar-time portfolio")
    portfolio, alpha = run_calendar_time(events, raw.panel, raw.factors, cfg)

    if verbose:
        print("Backtests")
    backtests, summary = run_backtests(events, raw.panel, raw.factors, cfg)

    if verbose:
        print("Cost sweep")
    curve, breakeven_close = cost_sweep(events, raw.panel, cfg, entry_timing="next_close")
    curve_open, breakeven_open = cost_sweep(events, raw.panel, cfg, entry_timing="next_open")
    curve = curve.merge(
        curve_open.rename(columns={"net_total_return_pct": "net_total_return_pct_next_open"}),
        on="round_trip_bps",
    ).rename(columns={"net_total_return_pct": "net_total_return_pct_next_close"})

    if verbose:
        print("Decay curve")
    decay_curve = decay.rolling_spread(
        events, "abdrift_d20", n_boot=max(cfg.n_bootstrap // 5, 1000), seed=cfg.random_seed
    )
    trend = decay.trend_test(decay_curve)

    results = StudyResults(
        events=events,
        audit=audit,
        tests=tests,
        calendar_time=portfolio,
        calendar_alpha=alpha,
        backtests=backtests,
        backtest_summary=summary,
        cost_curve=curve,
        breakeven_bps={"next_close": breakeven_close, "next_open": breakeven_open},
        decay_curve=decay_curve,
        decay_trend=trend,
        power=run_power(events, tests, "abdrift_d20"),
        timing=pipeline.timing_breakdown(events),
        decomposition=decompose_reaction(events, cfg),
    )
    write_results(results, cfg, results_dir=results_dir, output_dir=output_dir)
    return results


def write_results(
    results: StudyResults,
    cfg: StudyConfig,
    *,
    results_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Persist every result table under ``docs/results`` with provenance."""
    config.ensure_dirs()
    results_dir = results_dir or config.RESULTS_DIR
    output_dir = output_dir or config.OUTPUT_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {
        "inference_tests": results.tests,
        "calendar_time_alpha": results.calendar_alpha,
        "backtest_summary": results.backtest_summary,
        "cost_curve": results.cost_curve,
        "decay_rolling_spread": results.decay_curve,
        "power_analysis": results.power,
        "announcement_timing": results.timing,
        "reaction_decomposition": results.decomposition,
        "sample_audit": pd.DataFrame(
            [{"stage": k, "n": v} for k, v in results.audit.items()]
        ),
        "breakeven_cost": pd.DataFrame(
            [{"entry_timing": k, "breakeven_round_trip_bps": v} for k, v in results.breakeven_bps.items()]
        ),
    }
    written: list[Path] = []
    for name, frame in tables.items():
        path = results_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        config.write_provenance(
            path,
            command="python main.py study",
            data_source="SEC EDGAR (8-K item 2.02, XBRL EPS), Yahoo Finance prices, Ken French factors",
            as_of=cfg.end_date,
            extra={"n_events": int(len(results.events)), "rows": int(len(frame))},
        )
        written.append(path)

    daily = results.backtests["next_close"].daily if "next_close" in results.backtests else pd.DataFrame()
    if not daily.empty:
        path = output_dir / "backtest_daily_next_close.parquet"
        daily.to_parquet(path)
        written.append(path)
    return written
