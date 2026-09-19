"""A daily-marked long-beat/short-miss book.

The first version of this repo scored the strategy on 56 overlapping trade
returns and called the result a Sharpe ratio. It is not one: a Sharpe ratio is
a property of a return *series* sampled at a known frequency, and a set of
overlapping trade outcomes is neither.

So the book is marked every session instead. Every name inside its holding
window is carried at an equal share of gross exposure, the portfolio return is
computed daily, and drawdown, volatility and Sharpe all come off that series.
Costs are charged on the notional actually traded each day plus a borrow
accrual on the short leg, which is the only way to make the cost figure
respond to how concentrated the earnings calendar is.

Entry timing is a parameter rather than a convention, because the choice
changes what is being measured:

``next_open``
    Fill at the opening auction of the first session the market can react in.
    The earliest fill anyone could actually get. It forgoes the overnight gap,
    which is not fillable, but keeps the first session's move.
``next_close``
    Fill at that session's close. Forgoes the entire announcement reaction and
    earns only the drift that follows it. Conservative, and the closest thing
    to a pure test of post-announcement drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CostAssumptions, StudyConfig
from .costs import TRADING_DAYS_PER_YEAR, CostModel, break_even_cost_bps
from .events import entry_position_column
from .prices import PricePanel


@dataclass
class BacktestResult:
    """Daily series, per-trade log and headline metrics of one run."""

    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | str] = field(default_factory=dict)


def _position_windows(
    events: pd.DataFrame, hold_days: int, n_sessions: int, entry_timing: str
) -> pd.DataFrame:
    """Entry and exit session of every position, for one fill convention.

    The entry session comes from the column :func:`esa.events.entry_position_column`
    names, not from the reaction session, because a mid-session release cannot
    be filled at that session's opening auction.
    """
    tradeable = events[events["category"].isin(["Beat", "Miss"])].copy()
    tradeable["side"] = np.where(tradeable["category"] == "Beat", 1.0, -1.0)
    column = entry_position_column(entry_timing)
    if column not in tradeable.columns:  # panels built before the split
        column = "reaction_pos"
    tradeable["entry_pos"] = tradeable[column].astype(int)
    tradeable["exit_pos"] = tradeable["entry_pos"] + hold_days
    return tradeable[tradeable["exit_pos"] < n_sessions].reset_index(drop=True)


def _daily_returns(
    panel: PricePanel, entry_timing: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three return matrices: full session, open-to-close, and previous-close-to-open."""
    close = panel.close.to_numpy(dtype=float)
    open_ = panel.open.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        full = np.vstack([np.full((1, close.shape[1]), np.nan), close[1:] / close[:-1] - 1.0])
        open_to_close = close / open_ - 1.0
        prev_close_to_open = np.vstack(
            [np.full((1, close.shape[1]), np.nan), open_[1:] / close[:-1] - 1.0]
        )
    del entry_timing
    return full, open_to_close, prev_close_to_open


def _trade_log(
    positions: pd.DataFrame,
    panel: PricePanel,
    dates: pd.DatetimeIndex,
    model: CostModel,
    cfg: StudyConfig,
    entry_timing: str,
) -> pd.DataFrame:
    """Per-position record, priced entry to exit.

    This is a descriptive log, not the performance series. It prices each
    position buy-and-hold from its own fill to its own exit, so hit rate,
    average win and profit factor mean what a reader expects them to mean.

    It does not add up to the portfolio return, and should not: the book is
    equal-weighted across whatever is live and rebalanced daily, so a short
    held through a 1% daily decline earns +1% a day compounding, while the
    same short priced entry-to-exit earns slightly less. The daily series in
    ``BacktestResult.daily`` is the authoritative one.
    """
    prices = panel.open if entry_timing == "next_open" else panel.close
    matrix = prices.to_numpy(dtype=float)
    col = positions["col"].to_numpy()
    entry_price = matrix[positions["entry_pos"].to_numpy(), col]
    exit_price = matrix[positions["exit_pos"].to_numpy(), col]
    side = positions["side"].to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        gross = side * (exit_price / entry_price - 1.0)

    round_trip = model.round_trip_bps / 10_000.0
    borrow = model.borrow_bps_per_day / 10_000.0 * cfg.hold_days

    trades = positions[["ticker", "category", "side", "entry_pos", "exit_pos"]].copy()
    trades["entry_date"] = dates[trades["entry_pos"].to_numpy()]
    trades["exit_date"] = dates[trades["exit_pos"].to_numpy()]
    trades["entry_price"] = entry_price
    trades["exit_price"] = exit_price
    trades["gross_ret_pct"] = gross * 100.0
    trades["net_ret_pct"] = (gross - round_trip - np.where(side < 0, borrow, 0.0)) * 100.0
    return trades


def run_backtest(
    events: pd.DataFrame,
    panel: PricePanel,
    cfg: StudyConfig,
    *,
    costs: CostAssumptions | None = None,
    entry_timing: str | None = None,
) -> BacktestResult:
    """Mark the strategy daily and return series, trades and metrics."""
    entry_timing = entry_timing or cfg.entry_timing
    if entry_timing not in ("next_open", "next_close"):
        raise ValueError(f"unknown entry_timing: {entry_timing!r}")
    model = CostModel(costs or cfg.costs)

    dates = panel.dates
    n_sessions = len(dates)
    positions = _position_windows(events, cfg.hold_days, n_sessions, entry_timing)
    if positions.empty:
        return BacktestResult(pd.DataFrame(), pd.DataFrame(), {})

    col_of = {t: i for i, t in enumerate(panel.close.columns)}
    positions = positions[positions["ticker"].isin(col_of)].reset_index(drop=True)
    positions["col"] = positions["ticker"].map(col_of).astype(int)

    full, open_to_close, prev_close_to_open = _daily_returns(panel, entry_timing)

    opens_on: dict[int, list[int]] = {}
    for idx, entry in enumerate(positions["entry_pos"].to_numpy()):
        opens_on.setdefault(int(entry), []).append(idx)

    side = positions["side"].to_numpy()
    col = positions["col"].to_numpy()
    exit_pos = positions["exit_pos"].to_numpy()

    active: dict[int, float] = {}  # position id -> signed weight held into today
    gross = np.full(n_sessions, np.nan)
    net = np.full(n_sessions, np.nan)
    turnover = np.zeros(n_sessions)
    n_long = np.zeros(n_sessions, dtype=int)
    n_short = np.zeros(n_sessions, dtype=int)
    short_exposure = np.zeros(n_sessions)

    for t in range(n_sessions):
        opening = opens_on.get(t, [])
        holding = [pid for pid in active if exit_pos[pid] >= t]
        live = holding + opening
        if not live:
            active = {}
            continue

        weight = 1.0 / len(live)
        rets = np.zeros(len(live))
        for k, pid in enumerate(live):
            if pid in opening:
                # Entry session: an open fill earns the rest of the session, a
                # close fill earns nothing until tomorrow.
                r = open_to_close[t, col[pid]] if entry_timing == "next_open" else 0.0
            elif t == exit_pos[pid] and entry_timing == "next_open":
                r = prev_close_to_open[t, col[pid]]
            else:
                r = full[t, col[pid]]
            rets[k] = 0.0 if not np.isfinite(r) else r

        signed = side[live]
        day_gross = float(np.sum(weight * signed * rets))

        # Notional traded: the gap between yesterday's book, drifted by
        # today's returns, and today's equal-weight target — plus the unwind
        # of anything whose holding period ends tonight. Charging the unwind
        # on the exit session rather than the session after matters: a book
        # that empties completely would otherwise never pay to close.
        traded = 0.0
        for k, pid in enumerate(live):
            previous = active.get(pid, 0.0) * (1.0 + rets[k])
            traded += abs(weight * signed[k] - previous)
            if exit_pos[pid] == t:
                traded += abs(weight)

        short_weight = float(np.sum(np.abs(weight * signed[signed < 0])))
        day_net = day_gross - model.turnover_cost(traded) - model.borrow_cost(short_weight)

        gross[t] = day_gross
        net[t] = day_net
        turnover[t] = traded
        n_long[t] = int(np.sum(signed > 0))
        n_short[t] = int(np.sum(signed < 0))
        short_exposure[t] = short_weight

        active = {pid: weight * signed[k] for k, pid in enumerate(live) if exit_pos[pid] > t}

    daily = pd.DataFrame(
        {
            "gross_ret": gross,
            "net_ret": net,
            "turnover": turnover,
            "n_long": n_long,
            "n_short": n_short,
            "short_exposure": short_exposure,
        },
        index=dates,
    )
    traded_days = daily["net_ret"].notna()
    daily["equity_gross"] = (1.0 + daily["gross_ret"].fillna(0.0)).cumprod()
    daily["equity_net"] = (1.0 + daily["net_ret"].fillna(0.0)).cumprod()

    trades = _trade_log(positions, panel, dates, model, cfg, entry_timing)

    metrics = compute_metrics(daily.loc[traded_days], trades, panel, model)
    metrics["entry_timing"] = entry_timing
    return BacktestResult(daily=daily, trades=trades, metrics=metrics)


def compute_metrics(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    panel: PricePanel,
    model: CostModel,
    *,
    risk_free: pd.Series | None = None,
) -> dict[str, float | str]:
    """Headline performance statistics, all computed on daily marks.

    ``risk_free`` is the daily simple bill rate. When supplied, Sharpe and
    Sortino are computed on the excess return. The book is not dollar-neutral
    — the Beat leg has far more names than the Miss leg — so it is treated as
    a funded portfolio and the bill rate is subtracted.
    """
    if daily.empty:
        return {}

    net = daily["net_ret"].fillna(0.0)
    gross = daily["gross_ret"].fillna(0.0)
    equity = float((1.0 + net).prod())
    span = pd.Timestamp(daily.index[-1]) - pd.Timestamp(daily.index[0])
    years = max(span.days / 365.25, 1e-9)

    excess = net - risk_free.reindex(daily.index).fillna(0.0) if risk_free is not None else net
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if excess.std(ddof=1) > 0
        else float("nan")
    )
    downside = excess[excess < 0]
    sortino = (
        float(excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else float("nan")
    )

    curve = (1.0 + net).cumprod()
    peak = curve.cummax()
    drawdown = curve / peak - 1.0
    trough = pd.Timestamp(drawdown.idxmin())
    peak_date = pd.Timestamp(curve.loc[:trough].idxmax())

    wins = trades["net_ret_pct"] > 0
    gross_profit = float(trades.loc[wins, "net_ret_pct"].sum())
    gross_loss = float(-trades.loc[~wins, "net_ret_pct"].sum())

    # The benchmark's own risk statistics, on the same sessions. A long-short
    # book running 10% volatility and a long-only index running twice that are
    # not comparable on total return alone, and quoting only the return gap
    # would flatter whichever side happened to be less risky.
    bench = panel.close.get("SPY")
    bench_total = float("nan")
    bench_vol = float("nan")
    bench_sharpe = float("nan")
    bench_drawdown = float("nan")
    bench_cagr = float("nan")
    if bench is not None:
        window = bench.loc[daily.index[0] : daily.index[-1]].dropna()
        if len(window) > 1:
            bench_total = float(window.iloc[-1] / window.iloc[0] - 1.0) * 100.0
            bench_daily = window.pct_change().dropna()
            bench_vol = float(bench_daily.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) * 100.0
            bench_excess = bench_daily
            if risk_free is not None:
                bench_excess = bench_daily - risk_free.reindex(bench_daily.index).fillna(0.0)
            if bench_excess.std(ddof=1) > 0:
                bench_sharpe = float(
                    bench_excess.mean() / bench_excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
                )
            bench_curve = (1.0 + bench_daily).cumprod()
            bench_drawdown = float((bench_curve / bench_curve.cummax() - 1.0).min()) * 100.0
            bench_cagr = ((1.0 + bench_total / 100.0) ** (1.0 / years) - 1.0) * 100.0

    return {
        "n_trades": float(len(trades)),
        "n_long_trades": float((trades["side"] > 0).sum()),
        "n_short_trades": float((trades["side"] < 0).sum()),
        "total_return_pct": (equity - 1.0) * 100.0,
        "gross_total_return_pct": float((1.0 + gross).prod() - 1.0) * 100.0,
        "cagr_pct": (equity ** (1.0 / years) - 1.0) * 100.0,
        "ann_vol_pct": vol * 100.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": float(drawdown.min()) * 100.0,
        # Dates as ISO strings: the block is written straight to CSV, and a
        # dict of floats with two timestamps in it is awkward everywhere else.
        "max_drawdown_peak": peak_date.date().isoformat(),
        "max_drawdown_trough": trough.date().isoformat(),
        "hit_rate_pct": float(wins.mean()) * 100.0,
        "avg_win_pct": float(trades.loc[wins, "net_ret_pct"].mean())
        if wins.any()
        else float("nan"),
        "avg_loss_pct": float(trades.loc[~wins, "net_ret_pct"].mean())
        if (~wins).any()
        else float("nan"),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "ann_turnover_x": float(daily["turnover"].mean() * TRADING_DAYS_PER_YEAR),
        "exposure_pct": float((daily["n_long"] + daily["n_short"] > 0).mean()) * 100.0,
        "avg_positions": float((daily["n_long"] + daily["n_short"]).mean()),
        "round_trip_cost_bps": model.round_trip_bps,
        "benchmark_total_return_pct": bench_total,
        "benchmark_ann_vol_pct": bench_vol,
        "benchmark_sharpe": bench_sharpe,
        "benchmark_max_drawdown_pct": bench_drawdown,
        "benchmark_cagr_pct": bench_cagr,
        "years": years,
    }


def net_return_at_cost(
    events: pd.DataFrame,
    panel: PricePanel,
    cfg: StudyConfig,
    bps: float,
    *,
    entry_timing: str | None = None,
) -> float:
    """Total net return, in percent, at a given round-trip cost."""
    assumptions = CostAssumptions(
        commission_bps=bps / 2.0,
        half_spread_bps=0.0,
        impact_coef_bps=0.0,
        participation_rate=cfg.costs.participation_rate,
        borrow_bps_annual=cfg.costs.borrow_bps_annual,
    )
    result = run_backtest(events, panel, cfg, costs=assumptions, entry_timing=entry_timing)
    total = result.metrics.get("total_return_pct", float("nan"))
    return float(total) if isinstance(total, (int, float)) else float("nan")


def cost_sweep(
    events: pd.DataFrame,
    panel: PricePanel,
    cfg: StudyConfig,
    *,
    grid: np.ndarray | None = None,
    entry_timing: str | None = None,
) -> tuple[pd.DataFrame, float]:
    """Net total return across a grid of round-trip costs, plus the break-even.

    The break-even is found by bisection on the same evaluation the grid uses,
    so the marked point on the figure and the number in the README come from
    one calculation.
    """
    grid = (
        grid
        if grid is not None
        else np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200], dtype=float)
    )
    rows = [
        {
            "round_trip_bps": float(bps),
            "net_total_return_pct": net_return_at_cost(
                events, panel, cfg, float(bps), entry_timing=entry_timing
            ),
        }
        for bps in grid
    ]
    breakeven = break_even_cost_bps(
        lambda bps: net_return_at_cost(events, panel, cfg, bps, entry_timing=entry_timing)
    )
    return pd.DataFrame(rows), breakeven
