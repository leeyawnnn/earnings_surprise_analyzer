"""Backtest accounting, checked against arithmetic done by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import SESSIONS, make_panel
from esa.backtest import run_backtest
from esa.config import CostAssumptions, StudyConfig
from esa.events import build_event_panel

FREE = CostAssumptions(
    commission_bps=0.0, half_spread_bps=0.0, impact_coef_bps=0.0, borrow_bps_annual=0.0
)


def _two_trade_panel() -> tuple[object, pd.DataFrame]:
    """One beat and one miss, on disjoint dates, with arithmetic-friendly prices.

    ``AAA`` rises 1% a session for five sessions from session 10.
    ``BBB`` falls 1% a session for five sessions from session 30.
    Both announcements are after the close of the preceding session, so the
    reaction session is the one the move starts in.
    """
    n = len(SESSIONS)
    aaa = [100.0] * n
    for k in range(10, n):
        aaa[k] = 100.0 * 1.01 ** min(k - 9, 5)
    bbb = [100.0] * n
    for k in range(30, n):
        bbb[k] = 100.0 * 0.99 ** min(k - 29, 5)
    panel = make_panel(
        {"AAA": aaa, "BBB": bbb, "SPY": [100.0] * n},
        {"AAA": aaa, "BBB": bbb, "SPY": [100.0] * n},
    )
    events = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "accepted_utc": (SESSIONS[9] + pd.Timedelta(hours=17))
                .tz_localize("America/New_York")
                .tz_convert("UTC"),
                "sue": 2.5,
            },
            {
                "ticker": "BBB",
                "accepted_utc": (SESSIONS[29] + pd.Timedelta(hours=17))
                .tz_localize("America/New_York")
                .tz_convert("UTC"),
                "sue": -2.5,
            },
        ]
    )
    built = build_event_panel(events, panel, windows=(0, 5))
    built["category"] = ["Beat", "Miss"]
    return panel, built


class TestTwoTradeAccounting:
    """Every number below is worked out on paper in the assertions."""

    def test_next_close_entry_earns_the_drift_only(self) -> None:
        panel, events = _two_trade_panel()
        cfg = StudyConfig(hold_days=4, costs=FREE)
        result = run_backtest(events, panel, cfg, entry_timing="next_close")

        assert len(result.trades) == 2
        # AAA enters at the close of session 10 (already up 1%) and exits at
        # the close of session 14, by which time it has risen four more 1%
        # steps: 1.01^4 - 1.
        long_trade = result.trades[result.trades["ticker"] == "AAA"].iloc[0]
        assert long_trade["gross_ret_pct"] == pytest.approx((1.01**4 - 1) * 100, rel=1e-9)
        # BBB is shorted at the close of session 30 and falls four more 1%
        # steps, so the short earns 1 - 0.99^4.
        short_trade = result.trades[result.trades["ticker"] == "BBB"].iloc[0]
        assert short_trade["gross_ret_pct"] == pytest.approx((1 - 0.99**4) * 100, rel=1e-9)

    def test_equity_compounds_the_daily_marks(self) -> None:
        """The book is rebalanced daily, so a short earns +1% each -1% session.

        That is deliberately not the same as pricing the short entry to exit,
        which would give 1 - 0.99**4. The daily series is the authoritative
        one and the trade log is descriptive, so the two are allowed to differ
        and the test pins which is which.
        """
        panel, events = _two_trade_panel()
        cfg = StudyConfig(hold_days=4, costs=FREE)
        result = run_backtest(events, panel, cfg, entry_timing="next_close")
        assert result.metrics["total_return_pct"] == pytest.approx((1.01**8 - 1) * 100, rel=1e-9)

        short_trade = result.trades[result.trades["ticker"] == "BBB"].iloc[0]
        assert short_trade["gross_ret_pct"] == pytest.approx((1 - 0.99**4) * 100, rel=1e-9)

    def test_next_open_entry_captures_what_next_close_gives_up(self) -> None:
        """The two conventions differ by exactly the first session's move.

        AAA gaps from a close of 100 to an open of 104 and finishes the
        session at 110. An opening fill earns 110/104 - 1; a closing fill
        starts at 110 and earns nothing, because the stock is flat after that.
        """
        n = len(SESSIONS)
        closes = [100.0] * n
        opens = [100.0] * n
        opens[10] = 104.0
        for k in range(10, n):
            closes[k] = 110.0
            if k > 10:
                opens[k] = 110.0
        panel = make_panel({"AAA": closes, "SPY": [100.0] * n}, {"AAA": opens, "SPY": [100.0] * n})
        stamp = (
            (SESSIONS[9] + pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("UTC")
        )
        events = pd.DataFrame([{"ticker": "AAA", "accepted_utc": stamp, "sue": 2.0}])
        built = build_event_panel(events, panel, windows=(0, 5))
        built["category"] = "Beat"
        cfg = StudyConfig(hold_days=2, costs=FREE)

        close_entry = run_backtest(built, panel, cfg, entry_timing="next_close")
        open_entry = run_backtest(built, panel, cfg, entry_timing="next_open")
        assert close_entry.metrics["total_return_pct"] == pytest.approx(0.0, abs=1e-9)
        assert open_entry.metrics["total_return_pct"] == pytest.approx(
            (110.0 / 104.0 - 1) * 100, rel=1e-9
        )

    def test_simultaneous_positions_share_capital(self) -> None:
        """Two names in the book on the same day are half-weighted each.

        This is the property the first version of the repo approximated with a
        per-entry-date batch average; here it falls out of the daily marking.
        """
        n = len(SESSIONS)
        up = [100.0] * n
        for k in range(10, n):
            up[k] = 110.0
        flat = [100.0] * n
        panel = make_panel(
            {"AAA": up, "BBB": flat, "SPY": flat}, {"AAA": up, "BBB": flat, "SPY": flat}
        )
        stamp = (
            (SESSIONS[9] + pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("UTC")
        )
        events = pd.DataFrame(
            [
                {"ticker": "AAA", "accepted_utc": stamp, "sue": 2.0},
                {"ticker": "BBB", "accepted_utc": stamp, "sue": 2.0},
            ]
        )
        built = build_event_panel(events, panel, windows=(0, 5))
        built["category"] = "Beat"
        cfg = StudyConfig(hold_days=3, costs=FREE)
        result = run_backtest(built, panel, cfg, entry_timing="next_close")
        # AAA is flat after session 10 and BBB never moves, so with both
        # entering at the close of session 10 the book earns nothing.
        assert result.metrics["total_return_pct"] == pytest.approx(0.0, abs=1e-9)
        active = result.daily.loc[SESSIONS[11], ["n_long", "n_short"]]
        assert int(active["n_long"]) == 2

    def test_costs_reduce_return_by_the_round_trip(self) -> None:
        panel, events = _two_trade_panel()
        cfg_free = StudyConfig(hold_days=4, costs=FREE)
        costed = CostAssumptions(
            commission_bps=5.0, half_spread_bps=0.0, impact_coef_bps=0.0, borrow_bps_annual=0.0
        )
        cfg_cost = StudyConfig(hold_days=4, costs=costed)
        free = run_backtest(events, panel, cfg_free, entry_timing="next_close")
        paid = run_backtest(events, panel, cfg_cost, entry_timing="next_close")
        assert paid.metrics["total_return_pct"] < free.metrics["total_return_pct"]
        # Two positions, each opened and closed once at 5 bp a side, against a
        # book that holds one name at a time: 20 bp of total drag.
        drag = free.metrics["total_return_pct"] - paid.metrics["total_return_pct"]
        assert drag == pytest.approx(0.20, abs=0.03)

    def test_borrow_is_charged_on_the_short_leg_only(self) -> None:
        panel, events = _two_trade_panel()
        borrow = CostAssumptions(
            commission_bps=0.0, half_spread_bps=0.0, impact_coef_bps=0.0, borrow_bps_annual=252.0
        )
        cfg = StudyConfig(hold_days=4, costs=borrow)
        result = run_backtest(events, panel, cfg, entry_timing="next_close")
        daily = result.daily
        short_days = daily[daily["n_short"] > 0]
        long_days = daily[(daily["n_long"] > 0) & (daily["n_short"] == 0)]
        # 252 bp a year is exactly 1 bp a session on a fully short book.
        drag = (short_days["gross_ret"] - short_days["net_ret"]).mean()
        assert drag == pytest.approx(1e-4, rel=0.01)
        assert (long_days["gross_ret"] - long_days["net_ret"]).abs().max() == pytest.approx(0.0)


class TestMetrics:
    def test_sharpe_is_computed_on_daily_marks(self) -> None:
        panel, events = _two_trade_panel()
        cfg = StudyConfig(hold_days=4, costs=FREE)
        result = run_backtest(events, panel, cfg, entry_timing="next_close")
        traded = result.daily["net_ret"].dropna()
        # There is one daily observation per session with a live position,
        # not one per trade.
        assert len(traded) > len(result.trades)
        assert np.isfinite(result.metrics["sharpe"])

    def test_exposure_counts_days_with_a_position(self) -> None:
        panel, events = _two_trade_panel()
        cfg = StudyConfig(hold_days=4, costs=FREE)
        result = run_backtest(events, panel, cfg, entry_timing="next_close")
        assert result.metrics["exposure_pct"] == pytest.approx(100.0)
        assert result.metrics["n_trades"] == 2
        assert result.metrics["n_long_trades"] == 1
        assert result.metrics["n_short_trades"] == 1

    def test_drawdown_dates_bracket_the_trough(self) -> None:
        panel, events = _two_trade_panel()
        cfg = StudyConfig(hold_days=4, costs=FREE)
        result = run_backtest(events, panel, cfg, entry_timing="next_close")
        assert result.metrics["max_drawdown_peak"] <= result.metrics["max_drawdown_trough"]

    def test_unknown_entry_timing_is_rejected(self) -> None:
        panel, events = _two_trade_panel()
        with pytest.raises(ValueError, match="entry_timing"):
            run_backtest(events, panel, StudyConfig(), entry_timing="tomorrow_maybe")

    def test_in_line_events_are_not_traded(self) -> None:
        panel, events = _two_trade_panel()
        events = events.assign(category="In-Line")
        result = run_backtest(events, panel, StudyConfig(hold_days=4, costs=FREE))
        assert result.trades.empty
