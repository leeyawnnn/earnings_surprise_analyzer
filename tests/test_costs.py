"""Cost arithmetic and the break-even search."""

from __future__ import annotations

import pytest

from esa.config import CostAssumptions
from esa.costs import TRADING_DAYS_PER_YEAR, CostModel, break_even_cost_bps


class TestCostModel:
    def test_one_way_is_the_sum_of_its_parts(self) -> None:
        model = CostModel(
            CostAssumptions(
                commission_bps=0.5,
                half_spread_bps=1.0,
                impact_coef_bps=10.0,
                participation_rate=0.01,
            )
        )
        # Square-root impact at 1% participation is a tenth of the coefficient.
        assert model.impact_bps == pytest.approx(1.0)
        assert model.one_way_bps == pytest.approx(2.5)
        assert model.round_trip_bps == pytest.approx(5.0)

    def test_impact_grows_with_the_square_root_of_participation(self) -> None:
        """Quadrupling size doubles the impact charge, it does not quadruple it."""
        base = CostAssumptions(commission_bps=0, half_spread_bps=0, impact_coef_bps=10.0)
        small = CostModel(base).impact_bps if base.participation_rate else 0.0
        del small
        one = CostModel(CostAssumptions(impact_coef_bps=10.0, participation_rate=0.01)).impact_bps
        four = CostModel(CostAssumptions(impact_coef_bps=10.0, participation_rate=0.04)).impact_bps
        assert four == pytest.approx(2 * one)

    def test_borrow_accrues_daily(self) -> None:
        model = CostModel(CostAssumptions(borrow_bps_annual=252.0))
        assert model.borrow_bps_per_day == pytest.approx(1.0)
        assert model.borrow_cost(1.0) == pytest.approx(1e-4)
        assert model.borrow_cost(0.5) == pytest.approx(0.5e-4)

    def test_borrow_is_charged_on_magnitude_not_sign(self) -> None:
        model = CostModel(CostAssumptions(borrow_bps_annual=TRADING_DAYS_PER_YEAR * 1.0))
        assert model.borrow_cost(-0.4) == pytest.approx(model.borrow_cost(0.4))

    def test_turnover_cost_is_linear(self) -> None:
        model = CostModel(CostAssumptions(commission_bps=5.0, half_spread_bps=0, impact_coef_bps=0))
        assert model.turnover_cost(1.0) == pytest.approx(5e-4)
        assert model.turnover_cost(2.0) == pytest.approx(10e-4)

    def test_with_round_trip_bps_rebuilds_the_total_and_keeps_borrow(self) -> None:
        """The sweep varies total friction, not how it splits between sources."""
        model = CostModel(CostAssumptions(borrow_bps_annual=40.0)).with_round_trip_bps(30.0)
        assert model.round_trip_bps == pytest.approx(30.0)
        assert model.assumptions.borrow_bps_annual == pytest.approx(40.0)


class TestBreakEven:
    def test_finds_the_root_of_a_linear_edge(self) -> None:
        # A strategy earning 40% gross that loses 1% of return per basis point.
        assert break_even_cost_bps(lambda bps: 40.0 - bps) == pytest.approx(40.0, abs=0.1)

    def test_zero_when_the_strategy_loses_at_no_cost(self) -> None:
        assert break_even_cost_bps(lambda bps: -5.0 - bps) == 0.0

    def test_infinite_when_it_survives_the_whole_range(self) -> None:
        assert break_even_cost_bps(lambda bps: 1000.0 - bps, hi=100.0) == float("inf")

    def test_tolerance_is_respected(self) -> None:
        found = break_even_cost_bps(lambda bps: 17.3 - bps, tolerance=0.01)
        assert found == pytest.approx(17.3, abs=0.02)
