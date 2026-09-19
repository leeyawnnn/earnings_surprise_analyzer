"""Consensus handling, with the network replaced by a fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esa.consensus import MATCH_TOLERANCE_DAYS, attach_to_events, load_consensus, save_consensus


def _quotes(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "announced_date": pd.to_datetime([r[1] for r in rows]),
            "eps_estimate": [r[2] for r in rows],
            "eps_actual": [r[3] for r in rows],
            "yahoo_surprise_pct": np.nan,
        }
    )


def _events(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "announcement_date": pd.to_datetime([r[1] for r in rows]),
            "sue": 1.0,
        }
    )


class TestAttachToEvents:
    def test_matches_on_the_same_day(self) -> None:
        events = _events([("AAA", "2020-05-01")])
        out = attach_to_events(events, _quotes([("AAA", "2020-05-01", 1.00, 1.10)]))
        assert out["pct_surprise"].iloc[0] == pytest.approx(10.0)

    def test_tolerates_a_day_either_side(self) -> None:
        """Yahoo's calendar date and the EDGAR acceptance date can disagree.

        Usually because one is recorded in local time and the other in
        Eastern, which puts an after-hours release on either side of midnight.
        """
        for offset in (-MATCH_TOLERANCE_DAYS, 0, MATCH_TOLERANCE_DAYS):
            day = pd.Timestamp("2020-05-01") + pd.Timedelta(days=offset)
            out = attach_to_events(
                _events([("AAA", "2020-05-01")]),
                _quotes([("AAA", f"{day:%Y-%m-%d}", 1.00, 1.10)]),
            )
            assert out["pct_surprise"].notna().all(), offset

    def test_rejects_a_match_beyond_the_tolerance(self) -> None:
        out = attach_to_events(
            _events([("AAA", "2020-05-01")]),
            _quotes([("AAA", "2020-05-20", 1.00, 1.10)]),
        )
        assert out["pct_surprise"].isna().all()

    def test_does_not_match_across_tickers(self) -> None:
        out = attach_to_events(
            _events([("AAA", "2020-05-01")]),
            _quotes([("BBB", "2020-05-01", 1.00, 1.10)]),
        )
        assert out["pct_surprise"].isna().all()

    def test_unmatched_events_are_kept_not_dropped(self) -> None:
        """Otherwise the overlap silently becomes the sample."""
        events = _events([("AAA", "2020-05-01"), ("AAA", "2021-05-01")])
        out = attach_to_events(events, _quotes([("AAA", "2020-05-01", 1.00, 1.10)]))
        assert len(out) == 2
        assert out["pct_surprise"].notna().sum() == 1

    def test_near_zero_estimates_come_back_missing(self) -> None:
        out = attach_to_events(
            _events([("AAA", "2020-05-01")]),
            _quotes([("AAA", "2020-05-01", 0.002, 0.20)]),
        )
        assert out["pct_surprise"].isna().all()

    def test_empty_consensus_is_handled(self) -> None:
        events = _events([("AAA", "2020-05-01")])
        out = attach_to_events(events, pd.DataFrame())
        assert out["pct_surprise"].isna().all()
        assert len(out) == 1


class TestCache:
    def test_round_trips_with_a_provenance_sidecar(self, tmp_path) -> None:
        frame = _quotes([("AAA", "2020-05-01", 1.0, 1.1)])
        path = save_consensus(frame, tmp_path)
        assert path.with_suffix(path.suffix + ".meta.json").exists()
        back = load_consensus(tmp_path)
        assert len(back) == 1
        assert back["ticker"].iloc[0] == "AAA"

    def test_missing_cache_raises_with_the_command_to_run(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="fetch_consensus"):
            load_consensus(tmp_path)
