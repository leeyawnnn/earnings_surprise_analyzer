"""EDGAR parsing, with the network replaced by fixtures.

No test in this file opens a socket. The responses are trimmed copies of the
shapes EDGAR actually returns, which is what the parsing has to survive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from esa.sec import (
    QUARTER_DAYS,
    YEAR_DAYS,
    SecClient,
    first_reported,
    quarterly_eps,
)


class FakeClient(SecClient):
    """A client whose ``get_json`` serves a dictionary instead of a request."""

    def __init__(self, responses: dict[str, object], tmp_path: Path) -> None:
        super().__init__(cache_dir=tmp_path)
        self._responses = responses
        self.calls: list[str] = []

    def get_json(self, url, cache_key, *, use_cache=True):  # type: ignore[override]
        self.calls.append(cache_key)
        return self._responses.get(cache_key)  # type: ignore[return-value]


def _submissions_block(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    keys = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]
    return {k: [r.get(k, "") for r in rows] for k in keys}


class TestEarningsAnnouncements:
    def test_selects_only_item_2_02_eight_ks(self, tmp_path: Path) -> None:
        rows = [
            {
                "accessionNumber": "0000-1",
                "filingDate": "2020-02-03",
                "acceptanceDateTime": "2020-02-03T21:30:00.000Z",
                "form": "8-K",
                "items": "2.02,9.01",
            },
            # An 8-K about something else entirely.
            {
                "accessionNumber": "0000-2",
                "filingDate": "2020-02-10",
                "acceptanceDateTime": "2020-02-10T13:00:00.000Z",
                "form": "8-K",
                "items": "5.02",
            },
            # The 10-Q that follows the release weeks later.
            {
                "accessionNumber": "0000-3",
                "filingDate": "2020-02-20",
                "acceptanceDateTime": "2020-02-20T21:00:00.000Z",
                "form": "10-Q",
                "items": "",
            },
        ]
        client = FakeClient(
            {"submissions_0000000123": {"filings": {"recent": _submissions_block(rows)}}}, tmp_path
        )
        out = client.earnings_announcements(123)
        assert list(out["accession"]) == ["0000-1"]
        assert out["accepted_utc"].iloc[0] == pd.Timestamp("2020-02-03T21:30:00Z")

    def test_follows_pagination_into_older_history(self, tmp_path: Path) -> None:
        """The recent block holds a thousand filings; the rest live elsewhere.

        Reading only the recent block truncates an active filer's history at
        roughly a decade, which would silently shorten the study period.
        """
        recent = _submissions_block(
            [
                {
                    "accessionNumber": "new",
                    "filingDate": "2022-02-03",
                    "acceptanceDateTime": "2022-02-03T21:30:00.000Z",
                    "form": "8-K",
                    "items": "2.02",
                }
            ]
        )
        older = _submissions_block(
            [
                {
                    "accessionNumber": "old",
                    "filingDate": "2011-02-03",
                    "acceptanceDateTime": "2011-02-03T21:30:00.000Z",
                    "form": "8-K",
                    "items": "2.02",
                }
            ]
        )
        client = FakeClient(
            {
                "submissions_0000000123": {
                    "filings": {
                        "recent": recent,
                        "files": [{"name": "CIK0000000123-submissions-001.json"}],
                    }
                },
                "submissions_0000000123_001": older,
            },
            tmp_path,
        )
        out = client.earnings_announcements(123)
        assert sorted(out["accession"]) == ["new", "old"]

    def test_missing_company_returns_an_empty_frame(self, tmp_path: Path) -> None:
        client = FakeClient({}, tmp_path)
        assert client.earnings_announcements(999).empty

    def test_a_filer_with_no_items_column_is_handled(self, tmp_path: Path) -> None:
        block = {"accessionNumber": ["x"], "filingDate": ["2020-01-01"], "form": ["8-K"]}
        client = FakeClient({"submissions_0000000123": {"filings": {"recent": block}}}, tmp_path)
        assert client.earnings_announcements(123).empty


class TestEpsFacts:
    def test_falls_back_to_basic_when_diluted_is_absent(self, tmp_path: Path) -> None:
        basic = {
            "units": {
                "USD/shares": [
                    {
                        "start": "2020-01-01",
                        "end": "2020-03-31",
                        "val": 1.0,
                        "filed": "2020-05-01",
                        "form": "10-Q",
                        "fy": 2020,
                        "fp": "Q1",
                    }
                ]
            }
        }
        client = FakeClient({"eps_0000000123_EarningsPerShareBasic": basic}, tmp_path)
        out = client.eps_facts(123)
        assert len(out) == 1
        assert out["tag"].iloc[0] == "EarningsPerShareBasic"
        # Diluted is tried first.
        assert client.calls[0].endswith("EarningsPerShareDiluted")

    def test_no_data_anywhere_returns_empty(self, tmp_path: Path) -> None:
        assert FakeClient({}, tmp_path).eps_facts(123).empty


def _fact(start: str, end: str, val: float, filed: str) -> dict[str, object]:
    return {
        "cik": 1,
        "tag": "EarningsPerShareDiluted",
        "start": start,
        "end": end,
        "val": val,
        "filed": filed,
    }


def _facts(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame["days"] = (frame["end"] - frame["start"]).dt.days
    return frame


class TestFirstReported:
    def test_keeps_the_earliest_filing_for_a_period(self) -> None:
        """A restatement must not overwrite what was originally published.

        Companies re-tag the same quarter in later filings, in comparative
        columns and again after a restatement. Only the first version was
        available to trade on.
        """
        facts = _facts(
            [
                _fact("2020-01-01", "2020-03-31", 1.00, "2020-05-01"),
                _fact("2020-01-01", "2020-03-31", 0.80, "2021-05-01"),
            ]
        )
        out = first_reported(facts, *QUARTER_DAYS)
        assert len(out) == 1
        assert out["val"].iloc[0] == pytest.approx(1.00)
        assert out["filed"].iloc[0] == pd.Timestamp("2020-05-01")

    def test_rejects_durations_outside_the_window(self) -> None:
        facts = _facts(
            [
                _fact("2020-01-01", "2020-06-30", 2.0, "2020-08-01"),  # half year
                _fact("2020-01-01", "2020-03-31", 1.0, "2020-05-01"),  # quarter
            ]
        )
        assert len(first_reported(facts, *QUARTER_DAYS)) == 1
        assert first_reported(facts, *YEAR_DAYS).empty

    def test_empty_input(self) -> None:
        assert first_reported(pd.DataFrame(), *QUARTER_DAYS).empty


class TestQuarterlyEps:
    def _three_quarters_and_a_year(self) -> pd.DataFrame:
        return _facts(
            [
                _fact("2020-01-01", "2020-03-31", 1.0, "2020-05-01"),
                _fact("2020-04-01", "2020-06-30", 1.2, "2020-08-01"),
                _fact("2020-07-01", "2020-09-30", 1.3, "2020-11-01"),
                _fact("2020-01-01", "2020-12-31", 5.0, "2021-02-01"),
            ]
        )

    def test_derives_the_fourth_quarter_from_the_annual_figure(self) -> None:
        """Filers disclose only the full year in the 10-K, so Q4 is a residual."""
        out = quarterly_eps(self._three_quarters_and_a_year())
        assert len(out) == 4
        q4 = out[out["derived"]].iloc[0]
        assert q4["val"] == pytest.approx(5.0 - (1.0 + 1.2 + 1.3))
        assert q4["end"] == pd.Timestamp("2020-12-31")
        # It becomes public with the annual filing, so that is when it is known.
        assert q4["filed"] == pd.Timestamp("2021-02-01")

    def test_a_directly_tagged_fourth_quarter_wins(self) -> None:
        facts = pd.concat(
            [
                self._three_quarters_and_a_year(),
                _facts([_fact("2020-10-01", "2020-12-31", 1.55, "2021-02-01")]),
            ],
            ignore_index=True,
        )
        out = quarterly_eps(facts)
        q4 = out[out["end"] == pd.Timestamp("2020-12-31")].iloc[0]
        assert not q4["derived"]
        assert q4["val"] == pytest.approx(1.55)

    def test_an_incomplete_year_is_not_derived(self) -> None:
        """Two quarters inside an annual period cannot give a fourth."""
        facts = _facts(
            [
                _fact("2020-01-01", "2020-03-31", 1.0, "2020-05-01"),
                _fact("2020-04-01", "2020-06-30", 1.2, "2020-08-01"),
                _fact("2020-01-01", "2020-12-31", 5.0, "2021-02-01"),
            ]
        )
        out = quarterly_eps(facts)
        assert not out["derived"].any()
        assert len(out) == 2

    def test_no_facts_gives_a_typed_empty_frame(self) -> None:
        out = quarterly_eps(pd.DataFrame())
        assert out.empty
        assert "derived" in out.columns


class TestCaching:
    def test_a_cached_null_is_not_refetched(self, tmp_path: Path) -> None:
        """EDGAR's 404 is a real answer and is cached as one."""
        (tmp_path / "missing.json").write_text("null")
        client = SecClient(cache_dir=tmp_path)
        assert client.get_json("https://example.invalid", "missing") is None

    def test_a_cached_document_is_read_from_disk(self, tmp_path: Path) -> None:
        (tmp_path / "doc.json").write_text(json.dumps({"hello": "world"}))
        client = SecClient(cache_dir=tmp_path)
        assert client.get_json("https://example.invalid", "doc") == {"hello": "world"}
