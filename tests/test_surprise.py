"""Bucketing, percent surprise guards, and SUE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esa.surprise import (
    categorize,
    categorize_series,
    percent_surprise,
    standardized_unexpected_earnings,
)


class TestCategorize:
    """Boundaries are exclusive, so the three buckets never overlap."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5.0001, "Beat"),
            (5.0, "In-Line"),
            (0.0, "In-Line"),
            (-5.0, "In-Line"),
            (-5.0001, "Miss"),
            (1e9, "Beat"),
            (-1e9, "Miss"),
        ],
    )
    def test_exact_thresholds(self, value: float, expected: str) -> None:
        assert categorize(value, beat=5.0, miss=-5.0) == expected

    def test_missing_is_its_own_bucket(self) -> None:
        assert categorize(float("nan"), beat=5.0, miss=-5.0) == "Unknown"
        assert categorize(None, beat=5.0, miss=-5.0) == "Unknown"

    def test_series_matches_scalar(self) -> None:
        values = pd.Series([5.0001, 5.0, -5.0, -5.0001, np.nan])
        vectorised = categorize_series(values, beat=5.0, miss=-5.0)
        scalar = [categorize(v, beat=5.0, miss=-5.0) for v in values]
        assert list(vectorised) == scalar

    def test_sigma_thresholds(self) -> None:
        values = pd.Series([1.5, 1.0, 0.0, -1.0, -1.5])
        assert list(categorize_series(values, beat=1.0, miss=-1.0)) == [
            "Beat",
            "In-Line",
            "In-Line",
            "In-Line",
            "Miss",
        ]


class TestPercentSurprise:
    def test_positive_estimate(self) -> None:
        assert percent_surprise(1.10, 1.00) == pytest.approx(10.0)

    def test_negative_estimate_keeps_the_economic_sign(self) -> None:
        """A smaller loss than feared is a positive surprise.

        Dividing by the signed estimate would flip this to negative, which is
        the wrong sign for every loss-making quarter in a sample.
        """
        assert percent_surprise(-0.50, -1.00) == pytest.approx(50.0)
        assert percent_surprise(-1.50, -1.00) == pytest.approx(-50.0)

    def test_near_zero_estimate_is_not_a_number(self) -> None:
        """A one-cent estimate would otherwise produce a 2,000% surprise."""
        assert np.isnan(percent_surprise(0.20, 0.0))
        assert np.isnan(percent_surprise(0.20, 0.001))
        assert np.isnan(percent_surprise(0.20, -0.009))

    def test_just_past_the_denominator_floor(self) -> None:
        assert percent_surprise(0.02, 0.01) == pytest.approx(100.0)

    def test_missing_inputs(self) -> None:
        assert np.isnan(percent_surprise(np.nan, 1.0))
        assert np.isnan(percent_surprise(1.0, np.nan))

    def test_cap(self) -> None:
        assert percent_surprise(10.0, 0.05, cap=200.0) == pytest.approx(200.0)
        assert percent_surprise(-10.0, 0.05, cap=200.0) == pytest.approx(-200.0)


def _eps_panel(values: list[float], *, ticker: str = "AAA") -> pd.DataFrame:
    """Quarterly EPS on a clean calendar, filed 30 days after each period."""
    ends = pd.date_range("2010-03-31", periods=len(values), freq="QE")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "end": ends,
            "val": values,
            "filed": ends + pd.Timedelta(days=30),
        }
    )


class TestSUE:
    def test_constant_earnings_give_a_degenerate_scale(self) -> None:
        """With no variation in the seasonal difference, SUE is undefined.

        Returning NaN rather than dividing by zero matters: the alternative is
        an infinite SUE that lands in the extreme bucket of every figure.
        """
        panel = _eps_panel([1.0] * 20)
        out = standardized_unexpected_earnings(panel, history_quarters=8)
        assert out["sue"].isna().all()

    def test_hand_computed_value(self) -> None:
        """A single planted jump against a known-variance history.

        Quarters 0-3 are 1.0 and every later quarter grows by exactly 0.10
        year over year, so the seasonal difference is 0.10 with zero variance
        — until the final quarter jumps by 0.50 instead.
        """
        values = [1.0, 1.0, 1.0, 1.0]
        for i in range(4, 16):
            values.append(values[i - 4] + 0.10)
        panel = _eps_panel(values)
        out = standardized_unexpected_earnings(panel, history_quarters=8).set_index("end")
        # Every difference in the history window is 0.10, so sigma is zero and
        # SUE is undefined regardless of the numerator.
        assert out["sue"].isna().all()
        assert out["drift"].dropna().unique() == pytest.approx([0.10])

    def test_detects_a_jump_against_a_noisy_history(self) -> None:
        rng = np.random.default_rng(0)
        base = 1.0
        values = [base + 0.01 * i for i in range(4)]
        for i in range(4, 24):
            values.append(values[i - 4] + 0.10 + float(rng.normal(0, 0.02)))
        # Plant a large positive surprise in the last quarter.
        values.append(values[-4] + 1.0)
        panel = _eps_panel(values)
        out = standardized_unexpected_earnings(panel, history_quarters=8)
        assert out["sue"].iloc[-1] > 5.0

    def test_history_known_by_precedes_the_quarter_being_measured(self) -> None:
        """Every input to a forecast must already be on file.

        This column is what the pipeline's point-in-time guard compares the
        announcement date against, so it has to be the latest filing date
        among the lag and the whole history window, not just the lag.
        """
        rng = np.random.default_rng(11)
        values = [1.0 + 0.05 * i + float(rng.normal(0, 0.02)) for i in range(24)]
        panel = _eps_panel(values)
        out = standardized_unexpected_earnings(panel, history_quarters=8)
        usable = out.dropna(subset=["sue"])
        assert not usable.empty
        # The forecast for quarter t uses filings up to quarter t-1, which was
        # filed 30 days after a period ending one quarter earlier.
        assert (usable["history_known_by"] < usable["filed"]).all()
        assert (usable["history_known_by"] >= usable["end"] - pd.Timedelta(days=200)).all()

    def test_gap_in_the_seasonal_lag_is_rejected(self) -> None:
        """A missing quarter must not silently become a five-quarter lag."""
        values = [1.0 + 0.05 * i for i in range(24)]
        panel = _eps_panel(values)
        panel = panel.drop(index=10).reset_index(drop=True)
        out = standardized_unexpected_earnings(panel, history_quarters=8)
        gap_rows = out[(out["end"] > panel["end"].iloc[9]) & (out["end"] <= panel["end"].iloc[13])]
        assert gap_rows["seasonal_diff"].isna().any()

    def test_multiple_tickers_do_not_bleed_into_each_other(self) -> None:
        a = _eps_panel([1.0 + 0.05 * i for i in range(16)], ticker="AAA")
        b = _eps_panel([5.0 - 0.05 * i for i in range(16)], ticker="BBB")
        out = standardized_unexpected_earnings(pd.concat([a, b]), history_quarters=8)
        for ticker in ("AAA", "BBB"):
            subset = out[out["ticker"] == ticker].sort_values("end")
            manual = subset["val"].to_numpy()[4:] - subset["val"].to_numpy()[:-4]
            assert subset["seasonal_diff"].dropna().to_numpy() == pytest.approx(manual)

    def test_drift_can_be_switched_off(self) -> None:
        values = [1.0 + 0.05 * i for i in range(24)]
        rng = np.random.default_rng(3)
        values = [v + float(rng.normal(0, 0.03)) for v in values]
        panel = _eps_panel(values)
        with_drift = standardized_unexpected_earnings(panel, include_drift=True)
        without = standardized_unexpected_earnings(panel, include_drift=False)
        both = with_drift.dropna(subset=["sue"])
        assert not both.empty
        assert not np.allclose(both["sue"].to_numpy(), without.loc[both.index, "sue"].to_numpy())

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            standardized_unexpected_earnings(pd.DataFrame({"ticker": ["A"], "end": [1]}))
