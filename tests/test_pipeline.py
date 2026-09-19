"""The joins and the guards that decide which events enter the sample."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import SESSIONS, make_panel, utc
from esa import universe
from esa.pipeline import (
    MAX_ANNOUNCEMENT_LAG_DAYS,
    match_announcements,
    plausible_eps_mask,
)


def _quarters(ends: list[str], ticker: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "end": pd.to_datetime(ends),
            "val": np.arange(1.0, len(ends) + 1.0),
            "filed": pd.to_datetime(ends) + pd.Timedelta(days=45),
        }
    )


def _announcements(stamps: list[str], ticker: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "accession": [f"acc-{i}" for i in range(len(stamps))],
            "accepted_utc": [utc(s) for s in stamps],
        }
    )


class TestMatchAnnouncements:
    def test_matches_a_quarter_to_the_release_that_follows_it(self) -> None:
        """The event date is when the result became public, not when the books closed.

        This is the defect the first version of the study had: it used the
        fiscal period end, which precedes the release by two to four weeks, so
        every position was opened before the information existed.
        """
        eps = _quarters(["2020-03-31", "2020-06-30"])
        announced = _announcements(["2020-04-28 16:30", "2020-07-28 16:30"])
        out = match_announcements(eps, announced)
        assert len(out) == 2
        lags = (out["accepted_utc"].dt.tz_convert(None).dt.normalize() - out["end"]).dt.days
        assert lags.tolist() == [28, 28]

    def test_never_matches_a_release_before_the_period_ends(self) -> None:
        eps = _quarters(["2020-06-30"])
        announced = _announcements(["2020-04-28 16:30"])
        assert match_announcements(eps, announced).empty

    def test_a_release_beyond_the_tolerance_is_dropped(self) -> None:
        eps = _quarters(["2020-03-31"])
        late = pd.Timestamp("2020-03-31") + pd.Timedelta(days=MAX_ANNOUNCEMENT_LAG_DAYS + 20)
        announced = _announcements([f"{late:%Y-%m-%d} 16:30"])
        assert match_announcements(eps, announced).empty

    def test_one_filing_announces_one_quarter(self) -> None:
        """A gap in a company's 8-K history must not attach a stale quarter.

        With two quarters and only the later release on file, a forward join
        would point both at the same filing and credit the older quarter's
        surprise to a release that never mentioned it.
        """
        eps = _quarters(["2020-03-31", "2020-06-30"])
        announced = _announcements(["2020-07-28 16:30"])
        out = match_announcements(eps, announced)
        assert len(out) == 1
        assert out["end"].iloc[0] == pd.Timestamp("2020-06-30")

    def test_tickers_do_not_cross_match(self) -> None:
        eps = pd.concat([_quarters(["2020-03-31"], "AAA"), _quarters(["2020-03-31"], "BBB")])
        announced = _announcements(["2020-04-28 16:30"], "AAA")
        out = match_announcements(eps, announced)
        assert out["ticker"].tolist() == ["AAA"]


class TestPlausibleEps:
    def _panel(self) -> object:
        n = len(SESSIONS)
        return make_panel({"AAA": [50.0] * n, "SPY": [100.0] * n})

    def test_a_normal_quarter_passes(self) -> None:
        events = pd.DataFrame({"ticker": ["AAA"], "end": [SESSIONS[10]], "val": [1.25]})
        assert plausible_eps_mask(events, self._panel()).all()

    def test_a_whole_dollar_income_tag_is_rejected(self) -> None:
        """Some filers tag this concept with income, not income per share."""
        events = pd.DataFrame({"ticker": ["AAA"], "end": [SESSIONS[10]], "val": [120_000_000.0]})
        assert not plausible_eps_mask(events, self._panel()).any()

    def test_a_dual_class_mismatch_is_rejected(self) -> None:
        """Class A earnings against a Class B price is the same failure, smaller."""
        events = pd.DataFrame({"ticker": ["AAA"], "end": [SESSIONS[10]], "val": [7000.0]})
        assert not plausible_eps_mask(events, self._panel()).any()

    def test_a_high_priced_name_with_large_eps_survives(self) -> None:
        """A flat dollar cap would wrongly exclude these, which is why it is a ratio."""
        n = len(SESSIONS)
        panel = make_panel({"AAA": [7000.0] * n, "SPY": [100.0] * n})
        events = pd.DataFrame({"ticker": ["AAA"], "end": [SESSIONS[10]], "val": [120.0]})
        assert plausible_eps_mask(events, panel).all()

    def test_a_loss_is_judged_on_magnitude(self) -> None:
        events = pd.DataFrame({"ticker": ["AAA"], "end": [SESSIONS[10]], "val": [-1.0]})
        assert plausible_eps_mask(events, self._panel()).all()

    def test_an_unknown_ticker_is_not_treated_as_a_bad_tag(self) -> None:
        """A missing price is missing evidence; the price-window filter removes it later."""
        events = pd.DataFrame({"ticker": ["ZZZ"], "end": [SESSIONS[10]], "val": [1.0]})
        assert plausible_eps_mask(events, self._panel()).all()


class TestUniverseEligibility:
    def _members(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ["OLD", "NEW", "UNDATED"],
                "name": ["Old Co", "New Co", "Undated Co"],
                "sector": ["X", "Y", "Z"],
                "cik": [1, 2, 3],
                "date_added": pd.to_datetime(["1990-01-01", "2021-06-01", None]),
            }
        )

    def test_an_event_before_the_join_date_is_excluded(self) -> None:
        events = pd.DataFrame(
            {
                "ticker": ["OLD", "NEW"],
                "announcement_date": pd.to_datetime(["2015-01-01", "2015-01-01"]),
            }
        )
        assert universe.eligibility_mask(self._members(), events).tolist() == [True, False]

    def test_an_event_after_the_join_date_is_included(self) -> None:
        events = pd.DataFrame(
            {"ticker": ["NEW"], "announcement_date": pd.to_datetime(["2022-01-01"])}
        )
        assert universe.eligibility_mask(self._members(), events).all()

    def test_an_undated_member_is_treated_as_always_present(self) -> None:
        events = pd.DataFrame(
            {"ticker": ["UNDATED"], "announcement_date": pd.to_datetime(["1995-01-01"])}
        )
        assert universe.eligibility_mask(self._members(), events).all()

    def test_members_on_is_one_sided(self) -> None:
        """It removes backfill bias. It cannot remove survivorship bias."""
        members = self._members()
        assert universe.members_on(members, pd.Timestamp("2015-01-01")) == ["OLD", "UNDATED"]
        assert set(universe.members_on(members, pd.Timestamp("2022-01-01"))) == {
            "OLD",
            "NEW",
            "UNDATED",
        }

    def test_load_universe_rejects_a_frame_missing_columns(self, tmp_path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame({"ticker": ["A"], "date_added": ["2020-01-01"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            universe.load_universe(path)
