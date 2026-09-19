"""Trading frictions.

A backtest without costs is not a conservative estimate of a strategy, it is a
different strategy. This one turns its book over completely every holding
period, so the cost assumption is not a rounding correction — it decides the
sign of the answer.

Three charges are modelled:

*Spread and commission* are proportional to notional traded and are charged on
both legs of every round trip.

*Market impact* uses the square-root law: pushing size into a name moves it
against you roughly as the square root of the fraction of daily volume taken.
At the default 1% participation the charge is a tenth of the coefficient,
which is deliberately not the optimistic end of published calibrations.

*Stock-loan fees* accrue daily on the short leg only, at an annualised rate.
Index-constituent shorts are usually general collateral and cheap, but "cheap"
is an assumption with a number attached, so it has one here.

The most useful output is not the net Sharpe but :func:`break_even_cost_bps`:
the round-trip cost at which the edge disappears. A strategy that breaks even
at 8 bp is dead in large caps; one that breaks even at 150 bp has room.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CostAssumptions

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CostModel:
    """Per-side and per-day charges implied by a set of assumptions."""

    assumptions: CostAssumptions

    @property
    def impact_bps(self) -> float:
        """Square-root market impact at the assumed participation rate."""
        return self.assumptions.impact_coef_bps * float(
            np.sqrt(max(self.assumptions.participation_rate, 0.0))
        )

    @property
    def one_way_bps(self) -> float:
        """Cost of trading one dollar of notional, once."""
        a = self.assumptions
        return a.commission_bps + a.half_spread_bps + self.impact_bps

    @property
    def round_trip_bps(self) -> float:
        """Cost of opening and closing one dollar of notional."""
        return 2.0 * self.one_way_bps

    @property
    def borrow_bps_per_day(self) -> float:
        """Daily accrual of the stock-loan fee on short notional."""
        return self.assumptions.borrow_bps_annual / TRADING_DAYS_PER_YEAR

    def turnover_cost(self, turnover: float) -> float:
        """Cost, in return units, of trading ``turnover`` fraction of capital."""
        return turnover * self.one_way_bps / 10_000.0

    def borrow_cost(self, short_exposure: float) -> float:
        """One day's borrow cost, in return units, on ``short_exposure``."""
        return abs(short_exposure) * self.borrow_bps_per_day / 10_000.0

    def with_round_trip_bps(self, bps: float) -> CostModel:
        """A model whose round trip costs exactly ``bps``, borrow unchanged.

        Used for the break-even sweep, where the question is how large the
        total friction can get before the edge vanishes — not how it splits
        between spread, commission and impact.
        """
        half = bps / 2.0
        return CostModel(
            CostAssumptions(
                commission_bps=half,
                half_spread_bps=0.0,
                impact_coef_bps=0.0,
                participation_rate=self.assumptions.participation_rate,
                borrow_bps_annual=self.assumptions.borrow_bps_annual,
            )
        )


def break_even_cost_bps(
    evaluate: object,
    *,
    lo: float = 0.0,
    hi: float = 1000.0,
    tolerance: float = 0.05,
    max_iter: int = 60,
) -> float:
    """Round-trip cost in basis points at which total net return reaches zero.

    ``evaluate`` maps a round-trip cost in basis points to the strategy's net
    total return. Net return falls monotonically in cost, so a bisection is
    well posed. Returns ``0.0`` when the strategy does not clear even zero
    cost, and ``inf`` when it survives the whole search range.
    """
    fn = evaluate  # type: ignore[assignment]
    if fn(lo) <= 0:  # type: ignore[operator]
        return 0.0
    if fn(hi) > 0:  # type: ignore[operator]
        return float("inf")

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if fn(mid) > 0:  # type: ignore[operator]
            lo = mid
        else:
            hi = mid
        if hi - lo < tolerance:
            break
    return (lo + hi) / 2.0
