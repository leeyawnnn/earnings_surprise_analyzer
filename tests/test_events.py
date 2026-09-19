"""Announcement timing, return windows, and the look-ahead guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import SESSIONS, make_panel, utc
from esa.events import (
    AFTER_CLOSE,
    BEFORE_OPEN,
    INTRADAY,
    abnormal_return_matrix,
    assert_no_lookahead,
    build_event_panel,
    classify_one,
    classify_timing,
    entry_position_column,
    entry_timestamp,
    reaction_positions,
)


class TestClassifyTiming:
    """EDGAR stamps acceptance in UTC; the session boundary is in New York."""

    @pytest.mark.parametrize(
        ("eastern", "expected"),
        [
            ("2020-02-03 06:00", BEFORE_OPEN),
            ("2020-02-03 09:29", BEFORE_OPEN),
            ("2020-02-03 09:30", INTRADAY),
            ("2020-02-03 12:00", INTRADAY),
            ("2020-02-03 16:00", INTRADAY),
            ("2020-02-03 16:01", AFTER_CLOSE),
            ("2020-02-03 20:30", AFTER_CLOSE),
        ],
    )
    def test_boundaries(self, eastern: str, expected: str) -> None:
        assert classify_one(utc(eastern)) == expected

    def test_summer_and_winter_offsets_agree(self) -> None:
        """16:30 New York is after the close in both July and January.

        The UTC offset differs by an hour across daylight saving, so a naive
        comparison against the raw UTC timestamp gets one of these wrong.
        """
        assert classify_one(utc("2020-07-15 16:30")) == AFTER_CLOSE
        assert classify_one(utc("2020-01-15 16:30")) == AFTER_CLOSE

    def test_series_input(self) -> None:
        stamps = pd.Series([utc("2020-02-03 07:00"), utc("2020-02-03 17:00")])
        assert list(classify_timing(stamps)) == [BEFORE_OPEN, AFTER_CLOSE]


class TestReactionSession:
    """Which session first absorbs the news."""

    def test_before_open_reacts_same_session(self) -> None:
        panel = make_panel()
        day = SESSIONS[10]
        announced = pd.Series([day + pd.Timedelta(hours=7)])
        timing = pd.Series([BEFORE_OPEN])
        assert reaction_positions(announced, timing, panel)[0] == 10

    def test_after_close_reacts_next_session(self) -> None:
        panel = make_panel()
        day = SESSIONS[10]
        announced = pd.Series([day + pd.Timedelta(hours=17)])
        timing = pd.Series([AFTER_CLOSE])
        assert reaction_positions(announced, timing, panel)[0] == 11

    def test_after_close_on_a_friday_reacts_on_monday(self) -> None:
        panel = make_panel()
        friday = pd.Timestamp("2020-01-10")
        assert friday.dayofweek == 4
        pos = reaction_positions(
            pd.Series([friday + pd.Timedelta(hours=17)]), pd.Series([AFTER_CLOSE]), panel
        )[0]
        assert panel.dates[pos] == pd.Timestamp("2020-01-13")

    def test_release_on_a_non_session_reacts_on_the_next_one(self) -> None:
        """A Saturday release is absorbed by Monday whatever the clock says."""
        panel = make_panel()
        saturday = pd.Timestamp("2020-01-11")
        for timing in (BEFORE_OPEN, AFTER_CLOSE):
            pos = reaction_positions(
                pd.Series([saturday + pd.Timedelta(hours=8)]), pd.Series([timing]), panel
            )[0]
            assert panel.dates[pos] == pd.Timestamp("2020-01-13")


class TestLookaheadGuard:
    """The test that must fail if entry timing ever regresses."""

    def test_passes_when_fills_follow_the_announcement(self) -> None:
        announced = pd.Series([utc("2020-02-03 16:30")])
        sessions = pd.Series([pd.Timestamp("2020-02-04")])
        assert_no_lookahead(announced, sessions, "next_open")
        assert_no_lookahead(announced, sessions, "next_close")

    def test_rejects_a_fill_on_the_announcement_session(self) -> None:
        """An after-close release cannot be traded at that session's close."""
        announced = pd.Series([utc("2020-02-03 16:30")])
        sessions = pd.Series([pd.Timestamp("2020-02-03")])
        with pytest.raises(AssertionError, match="look-ahead"):
            assert_no_lookahead(announced, sessions, "next_close")

    def test_rejects_a_fill_before_the_announcement(self) -> None:
        announced = pd.Series([utc("2020-02-03 16:30")])
        sessions = pd.Series([pd.Timestamp("2020-02-03")])
        with pytest.raises(AssertionError, match="look-ahead"):
            assert_no_lookahead(announced, sessions, "next_open")

    def test_open_fill_after_a_before_open_release_is_allowed(self) -> None:
        announced = pd.Series([utc("2020-02-03 07:00")])
        sessions = pd.Series([pd.Timestamp("2020-02-03")])
        assert_no_lookahead(announced, sessions, "next_open")

    def test_whole_panel_is_free_of_lookahead(self) -> None:
        """The guard applied the way the pipeline applies it, over many events.

        Includes mid-session releases, which are the case that separates the
        reacting session from the first session with a fillable open.
        """
        panel = make_panel()
        events = _mixed_events()
        built = build_event_panel(events, panel, windows=(0, 5))
        assert set(built["timing"]) == {BEFORE_OPEN, INTRADAY, AFTER_CLOSE}
        for timing in ("next_open", "next_close"):
            sessions = panel.dates[built[entry_position_column(timing)].to_numpy()]
            assert_no_lookahead(built["accepted_utc"], pd.Series(sessions), timing)

    def test_midsession_release_cannot_be_filled_at_that_open(self) -> None:
        """A 10:00 release is tradeable at that day's close, not its open."""
        panel = make_panel()
        events = pd.DataFrame(
            [{"ticker": "AAA", "accepted_utc": utc("2020-01-15 10:00"), "sue": 1.0}]
        )
        row = build_event_panel(events, panel, windows=(0, 5)).iloc[0]
        assert row["timing"] == INTRADAY
        assert panel.dates[row["reaction_pos"]] == pd.Timestamp("2020-01-15")
        assert panel.dates[row["entry_pos_close"]] == pd.Timestamp("2020-01-15")
        assert panel.dates[row["entry_pos_open"]] == pd.Timestamp("2020-01-16")

    def test_guard_catches_an_off_by_one_regression(self) -> None:
        """Simulate someone 'simplifying' after-close handling away."""
        panel = make_panel()
        events = _mixed_events()
        built = build_event_panel(events, panel, windows=(0, 5))
        regressed = panel.dates[built["reaction_pos"].to_numpy() - 1]
        with pytest.raises(AssertionError, match="look-ahead"):
            assert_no_lookahead(built["accepted_utc"], pd.Series(regressed), "next_close")


def _mixed_events() -> pd.DataFrame:
    rows = []
    for i, offset in enumerate([7, 10, 17, 21]):
        for day in (5, 12, 19, 26):
            rows.append(
                {
                    "ticker": "AAA",
                    "accepted_utc": (SESSIONS[day] + pd.Timedelta(hours=offset))
                    .tz_localize("America/New_York")
                    .tz_convert("UTC"),
                    "sue": float(i - 1.5),
                }
            )
    return pd.DataFrame(rows)


class TestEntryTimestamp:
    def test_open_and_close_map_to_the_right_instants(self) -> None:
        session = pd.Timestamp("2020-06-15")
        assert entry_timestamp(session, "next_open") == utc("2020-06-15 09:30")
        assert entry_timestamp(session, "next_close") == utc("2020-06-15 16:00")


class TestReturnWindows:
    """Hand-computed returns on a price path with known steps."""

    def _panel(self) -> tuple[object, int]:
        n = len(SESSIONS)
        closes = [100.0] * n
        opens = [100.0] * n
        # Announcement accepted after the close of session 9, so session 10
        # is the reaction session. Gap to 104, close of the reaction session
        # at 110, then +1 per session for five sessions.
        opens[10] = 104.0
        closes[10] = 110.0
        for k in range(1, 6):
            opens[10 + k] = 110.0 + k
            closes[10 + k] = 110.0 + k
        panel = make_panel({"AAA": closes, "SPY": [100.0] * n}, {"AAA": opens, "SPY": [100.0] * n})
        return panel, 10

    def test_day_zero_and_gap_decomposition(self) -> None:
        panel, _ = self._panel()
        events = pd.DataFrame(
            [{"ticker": "AAA", "accepted_utc": utc("2020-01-15 17:00"), "sue": 2.0}]
        )
        built = build_event_panel(events, panel, windows=(0, 5))
        assert len(built) == 1
        row = built.iloc[0]
        assert row["reaction_date"] == SESSIONS[10]
        # 100 -> 110 over the reaction session.
        assert row["ret_d0"] == pytest.approx(10.0)
        # 100 -> 104 overnight, 104 -> 110 during the session.
        assert row["gap_ret"] == pytest.approx(4.0)
        assert row["open_to_close_ret"] == pytest.approx(110.0 / 104.0 * 100 - 100)
        # The two legs compose back to the whole day.
        composed = (1 + row["gap_ret"] / 100) * (1 + row["open_to_close_ret"] / 100) - 1
        assert composed * 100 == pytest.approx(row["ret_d0"])

    def test_five_day_window_and_drift_split(self) -> None:
        panel, _ = self._panel()
        events = pd.DataFrame(
            [{"ticker": "AAA", "accepted_utc": utc("2020-01-15 17:00"), "sue": 2.0}]
        )
        row = build_event_panel(events, panel, windows=(0, 5)).iloc[0]
        # Pre-announcement close 100 to session 15 close 115.
        assert row["ret_d5"] == pytest.approx(15.0)
        # From the reaction close of 110 to 115.
        assert row["drift_d5"] == pytest.approx(110 / 110 * (115 / 110 - 1) * 100)
        assert row["drift_d5"] == pytest.approx(4.545454, abs=1e-5)

    def test_market_adjustment_subtracts_the_benchmark(self) -> None:
        n = len(SESSIONS)
        closes = {"AAA": [100.0] * n, "SPY": [100.0] * n}
        for k in range(10, n):
            closes["AAA"][k] = 110.0
            closes["SPY"][k] = 104.0
        panel = make_panel(closes, closes)
        events = pd.DataFrame(
            [{"ticker": "AAA", "accepted_utc": utc("2020-01-15 17:00"), "sue": 2.0}]
        )
        row = build_event_panel(events, panel, windows=(0, 5)).iloc[0]
        assert row["ret_d0"] == pytest.approx(10.0)
        assert row["abret_d0"] == pytest.approx(6.0)

    def test_events_without_room_for_the_window_are_dropped(self) -> None:
        panel = make_panel()
        late = SESSIONS[-2] + pd.Timedelta(hours=17)
        events = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "accepted_utc": late.tz_localize("America/New_York").tz_convert("UTC"),
                    "sue": 1.0,
                }
            ]
        )
        assert build_event_panel(events, panel, windows=(0, 20)).empty

    def test_unknown_ticker_is_dropped(self) -> None:
        panel = make_panel()
        events = pd.DataFrame(
            [{"ticker": "ZZZ", "accepted_utc": utc("2020-01-15 17:00"), "sue": 1.0}]
        )
        assert build_event_panel(events, panel, windows=(0, 5)).empty

    def test_season_label_follows_the_announcement_date(self) -> None:
        panel = make_panel()
        events = pd.DataFrame(
            [{"ticker": "AAA", "accepted_utc": utc("2020-01-15 17:00"), "sue": 1.0}]
        )
        assert build_event_panel(events, panel, windows=(0, 5)).iloc[0]["season"] == "2020Q1"


class TestAbnormalReturnMatrix:
    def test_path_matches_the_window_columns(self) -> None:
        n = len(SESSIONS)
        closes = [100.0] * n
        for k in range(10, n):
            closes[k] = 100.0 + (k - 9)
        panel = make_panel({"AAA": closes, "SPY": [100.0] * n})
        events = pd.DataFrame(
            [{"ticker": "AAA", "accepted_utc": utc("2020-01-15 17:00"), "sue": 1.0}]
        )
        built = build_event_panel(events, panel, windows=(0, 5))
        matrix = abnormal_return_matrix(built, panel, max_days=5, from_reaction_close=True)
        assert matrix.shape == (1, 6)
        assert matrix[0, 0] == pytest.approx(0.0)
        assert matrix[0, 5] == pytest.approx(built.iloc[0]["abdrift_d5"])

    def test_from_pre_announcement_close_includes_the_jump(self) -> None:
        n = len(SESSIONS)
        closes = [100.0] * n
        for k in range(10, n):
            closes[k] = 110.0
        panel = make_panel({"AAA": closes, "SPY": [100.0] * n})
        events = pd.DataFrame(
            [{"ticker": "AAA", "accepted_utc": utc("2020-01-15 17:00"), "sue": 1.0}]
        )
        built = build_event_panel(events, panel, windows=(0, 5))
        matrix = abnormal_return_matrix(built, panel, max_days=5, from_reaction_close=False)
        assert matrix[0, 0] == pytest.approx(10.0)
        assert np.allclose(matrix[0], 10.0)
