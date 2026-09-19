"""End to end, on the committed sample.

The unit tests pin individual behaviours against prices invented to make the
arithmetic checkable. This one runs the real pipeline over real filings and
real prices — twenty-four companies' worth — and asserts the properties that
have to hold whatever the data says.

It is also the test that would notice a change in an upstream file format,
since the committed sample is a copy of what the downloaders actually wrote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esa import config, pipeline, study
from esa.config import StudyConfig
from esa.events import assert_no_lookahead, entry_position_column
from esa.plotting import figures

pytestmark = pytest.mark.skipif(
    not (config.SAMPLE_DIR / "sec" / "announcements.parquet").exists(),
    reason="committed sample is missing; run scripts/build_sample.py",
)

#: Small enough to keep the suite quick, large enough for a stable p-value.
SAMPLE_CONFIG = StudyConfig(n_bootstrap=200, event_start_date="2013-01-01")


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> study.StudyResults:
    out = tmp_path_factory.mktemp("study")
    return study.run_study(
        SAMPLE_CONFIG,
        verbose=False,
        cache_dir=config.SAMPLE_DIR,
        results_dir=out / "results",
        output_dir=out / "output",
    )


class TestSampleStudy:
    def test_produces_a_usable_panel(self, results: study.StudyResults) -> None:
        events = results.events
        assert len(events) > 500
        assert events["ticker"].nunique() >= 15
        assert events["season"].nunique() >= 40
        assert set(events["category"]) <= {"Beat", "In-Line", "Miss", "Unknown"}

    def test_no_event_precedes_its_own_announcement(self, results: study.StudyResults) -> None:
        """The look-ahead guard, run over real filings rather than fixtures."""
        panel = pipeline.load_raw(config.SAMPLE_DIR).panel
        for timing in ("next_open", "next_close"):
            sessions = panel.dates[results.events[entry_position_column(timing)].to_numpy()]
            assert_no_lookahead(results.events["accepted_utc"], pd.Series(sessions), timing)

    def test_every_forecast_input_predates_its_announcement(
        self, results: study.StudyResults
    ) -> None:
        """The point-in-time guarantee, stated as an assertion.

        If this fails the surprise measure is using numbers that were not
        public when the release landed, which is the defect the whole data
        path exists to avoid.
        """
        events = results.events
        assert (events["history_known_by"] < events["announcement_date"]).all()

    def test_the_sample_audit_only_ever_shrinks(self, results: study.StudyResults) -> None:
        """Each filter is a filter. A stage that grows the sample is a join bug."""
        stages = list(results.audit.items())
        counts = [n for name, n in stages if name != "with_sue"]
        assert counts == sorted(counts, reverse=True), stages

    def test_announcement_timing_is_plausible(self, results: study.StudyResults) -> None:
        """Most US companies report before the open or after the close."""
        shares = results.timing.set_index("timing")["share_pct"]
        assert shares.get("before_open", 0) + shares.get("after_close", 0) > 85
        assert 0 < shares.get("intraday", 0) < 15


class TestInferenceOutputs:
    def test_naive_and_corrected_tests_share_a_point_estimate(
        self, results: study.StudyResults
    ) -> None:
        """They disagree about precision, never about the effect itself."""
        for metric, group in results.tests.groupby("metric"):
            estimates = group["estimate"].dropna().round(9).unique()
            assert len(estimates) == 1, (metric, estimates)

    def test_clustered_standard_errors_are_wider_on_average(
        self, results: study.StudyResults
    ) -> None:
        """The correction costs precision, on average.

        Not in every single metric: a bootstrap standard error is itself an
        estimate, and on a twenty-four company sample with a few hundred draws
        an individual one can land just below its naive counterpart. The claim
        that has to hold is about the average, with a floor that would catch a
        correction wired up backwards.
        """
        naive = results.tests[results.tests["method"].str.startswith("welch")]
        clustered = results.tests[results.tests["method"].str.startswith("block")]
        merged = naive.merge(clustered, on="metric", suffixes=("_naive", "_cluster"))
        ratio = merged["std_error_cluster"] / merged["std_error_naive"]
        assert ratio.mean() > 1.0
        assert ratio.min() > 0.8

    def test_p_values_are_in_range(self, results: study.StudyResults) -> None:
        p = results.tests["p_value"].dropna()
        assert ((p >= 0) & (p <= 1)).all()

    def test_calendar_time_portfolio_is_almost_always_invested(
        self, results: study.StudyResults
    ) -> None:
        """Inside the study window the book is rarely empty.

        Measured over the event window, not over the whole price calendar: the
        panel starts in 2009 to give the surprise measure its warm-up, and
        nothing is held during those first years by construction.

        The threshold is loose because coverage is a function of universe
        size. Twenty-four companies leave real gaps between reporting seasons;
        the full five hundred leave none, and the published run is invested on
        every session.
        """
        events = results.events
        window = results.calendar_time.loc[
            events["announcement_date"].min() : events["announcement_date"].max()
        ]
        held = window["n_long"] + window["n_short"]
        assert (held > 0).mean() > 0.7


class TestBacktestOutputs:
    def test_both_entry_conventions_report(self, results: study.StudyResults) -> None:
        assert set(results.backtest_summary["entry_timing"]) == {"next_open", "next_close"}

    def test_net_never_beats_gross(self, results: study.StudyResults) -> None:
        summary = results.backtest_summary
        assert (summary["total_return_pct"] <= summary["gross_total_return_pct"]).all()

    def test_cost_curve_falls_monotonically(self, results: study.StudyResults) -> None:
        curve = results.cost_curve.sort_values("round_trip_bps")
        for column in ("net_total_return_pct_next_close", "net_total_return_pct_next_open"):
            assert curve[column].diff().dropna().le(1e-9).all(), column

    def test_break_even_sits_where_the_curve_crosses_zero(
        self, results: study.StudyResults
    ) -> None:
        curve = results.cost_curve.sort_values("round_trip_bps")
        for timing, column in (
            ("next_close", "net_total_return_pct_next_close"),
            ("next_open", "net_total_return_pct_next_open"),
        ):
            point = results.breakeven_bps[timing]
            if not np.isfinite(point) or point <= 0:
                continue
            above = curve[curve["round_trip_bps"] <= point][column]
            below = curve[curve["round_trip_bps"] >= point][column]
            assert above.iloc[-1] >= -1e-6
            assert below.iloc[0] <= 1e-6


class TestFigures:
    def test_every_figure_renders(
        self, results: study.StudyResults, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """A figure that errors must fail the build, not disappear quietly."""
        out = tmp_path_factory.mktemp("figures")
        panel = pipeline.load_raw(config.SAMPLE_DIR).panel
        written = figures.build_all(results, panel, out)
        assert len(written) == 9
        for path in written:
            assert path.exists()
            assert path.stat().st_size > 5_000
            assert path.with_suffix(path.suffix + ".meta.json").exists()

    def test_results_tables_are_written_with_provenance(
        self, results: study.StudyResults, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        out = tmp_path_factory.mktemp("results")
        written = study.write_results(results, SAMPLE_CONFIG, results_dir=out, output_dir=out)
        assert written
        for path in written:
            if path.suffix != ".csv":
                continue
            meta = path.with_suffix(path.suffix + ".meta.json")
            assert meta.exists(), path
            assert "git_commit" in meta.read_text()
