"""Corrected inference: it must find a planted effect and not invent one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esa.inference import (
    SeasonStats,
    block_bootstrap_spread,
    calendar_time_alpha,
    calendar_time_returns,
    effective_sample_size,
    intra_cluster_correlation,
    required_sample_size,
    welch_spread,
    wild_cluster_bootstrap,
)

N_BOOT = 1500


def synthetic_events(
    *,
    effect: float,
    season_sd: float = 0.0,
    season_spread_sd: float = 0.0,
    event_sd: float = 5.0,
    n_seasons: int = 40,
    per_season: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Events with a known effect and two distinct kinds of season-level noise.

    ``season_sd`` is a shock common to every event in a season — a market move
    that lifts beats and misses alike. It makes the *level* of returns
    correlated within a season, but it cancels out of a long-short spread, so
    on its own it does not make the naive test wrong about the spread.

    ``season_spread_sd`` is a shock to the spread itself: the gap between
    beats and misses is wider in some seasons than others. This is the one
    that matters, and it is what the real data has, because how strongly the
    market underreacts is a property of the regime.

    Keeping the two separate is the point. Reaching for the first when the
    second is what breaks the test is an easy way to convince yourself a
    correction does nothing.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seasons):
        level_shock = rng.normal(0, season_sd)
        spread_shock = rng.normal(0, season_spread_sd)
        for i in range(per_season):
            bucket = "Beat" if i % 2 == 0 else "Miss"
            sign = 1.0 if bucket == "Beat" else -1.0
            mean = sign * (effect + spread_shock) / 2
            rows.append(
                {
                    "season": f"{2000 + s // 4}Q{s % 4 + 1}",
                    "category": bucket,
                    "value": mean + level_shock + rng.normal(0, event_sd),
                }
            )
    return pd.DataFrame(rows)


class TestSeasonStats:
    def test_full_resample_reproduces_the_pooled_mean(self) -> None:
        events = synthetic_events(effect=1.0, season_sd=1.0)
        stats_ = SeasonStats.build(events, "value")
        direct = (
            events.loc[events["category"] == "Beat", "value"].mean()
            - events.loc[events["category"] == "Miss", "value"].mean()
        )
        assert stats_.spread(np.arange(stats_.n_clusters)) == pytest.approx(direct)

    def test_duplicated_clusters_weight_by_count(self) -> None:
        """The estimator is a ratio of sums, so a repeated season counts twice."""
        events = synthetic_events(effect=2.0, season_sd=0.0, n_seasons=4, per_season=10)
        stats_ = SeasonStats.build(events, "value")
        doubled = stats_.spread(np.array([0, 0, 1, 1]))
        manual = events[events["season"].isin(sorted(events["season"].unique())[:2])]
        expected = (
            manual.loc[manual["category"] == "Beat", "value"].mean()
            - manual.loc[manual["category"] == "Miss", "value"].mean()
        )
        assert doubled == pytest.approx(expected)


class TestPlantedEffect:
    """A real effect, with clustering present, must still be detected."""

    def test_all_three_tests_find_a_large_effect(self) -> None:
        events = synthetic_events(effect=3.0, season_sd=1.0, season_spread_sd=0.5, seed=1)
        for result in (
            welch_spread(events, "value"),
            block_bootstrap_spread(events, "value", n_boot=N_BOOT, seed=1),
            wild_cluster_bootstrap(events, "value", n_boot=N_BOOT, seed=1),
        ):
            assert result.estimate == pytest.approx(3.0, abs=0.6)
            assert result.p_value < 0.05, result.method


class TestNullIsNotRejected:
    """No effect, but strong season-level correlation: the trap."""

    def test_naive_test_over_rejects_under_clustering(self) -> None:
        """Across many draws of a true null, Welch rejects far too often.

        This is the concrete statement of why the repo does not report a
        Welch p-value on its own. The size of a 5% test should be about 5%.
        """
        rejects_naive = 0
        rejects_cluster = 0
        trials = 40
        for seed in range(trials):
            events = synthetic_events(
                effect=0.0,
                season_spread_sd=3.0,
                n_seasons=20,
                per_season=40,
                seed=100 + seed,
            )
            if welch_spread(events, "value").p_value < 0.05:
                rejects_naive += 1
            if block_bootstrap_spread(events, "value", n_boot=400, seed=seed).p_value < 0.05:
                rejects_cluster += 1

        assert rejects_naive > rejects_cluster
        assert rejects_naive / trials > 0.20
        assert rejects_cluster / trials < 0.20

    def test_corrected_tests_accept_a_clean_null(self) -> None:
        events = synthetic_events(effect=0.0, season_sd=1.0, season_spread_sd=1.0, seed=7)
        for result in (
            block_bootstrap_spread(events, "value", n_boot=N_BOOT, seed=7),
            wild_cluster_bootstrap(events, "value", n_boot=N_BOOT, seed=7),
        ):
            assert result.p_value > 0.05, result.method


class TestStandardErrorsWiden:
    def test_clustered_errors_exceed_the_naive_one(self) -> None:
        """More shared noise must mean less claimed precision, not more."""
        events = synthetic_events(effect=1.0, season_spread_sd=3.0, seed=3)
        naive = welch_spread(events, "value")
        clustered = block_bootstrap_spread(events, "value", n_boot=N_BOOT, seed=3)
        assert clustered.std_error > naive.std_error

    def test_a_common_shock_cancels_out_of_the_spread(self) -> None:
        """A season-wide level move lifts both legs, so the spread is unaffected.

        Documenting this is as useful as the correction itself: it is the
        reason the corrected p-values in this repo are not catastrophically
        larger than the naive ones.
        """
        events = synthetic_events(effect=1.0, season_sd=5.0, seed=4)
        naive = welch_spread(events, "value")
        clustered = block_bootstrap_spread(events, "value", n_boot=N_BOOT, seed=4)
        assert clustered.std_error == pytest.approx(naive.std_error, rel=0.35)


class TestIntraClusterCorrelation:
    def test_zero_when_seasons_share_nothing(self) -> None:
        events = synthetic_events(effect=0.0, seed=5)
        assert intra_cluster_correlation(events, "value") < 0.02

    def test_rises_with_shared_variance(self) -> None:
        low = intra_cluster_correlation(synthetic_events(effect=0, season_sd=0.5, seed=6), "value")
        high = intra_cluster_correlation(synthetic_events(effect=0, season_sd=5.0, seed=6), "value")
        assert high > low


class TestPower:
    def test_clustering_inflates_the_required_sample(self) -> None:
        independent = required_sample_size(1.0, 10.0, icc=0.0, cluster_size=50)
        clustered = required_sample_size(1.0, 10.0, icc=0.1, cluster_size=50)
        assert clustered > independent * 5

    def test_zero_effect_needs_an_infinite_sample(self) -> None:
        assert required_sample_size(0.0, 10.0, icc=0.0, cluster_size=10) == float("inf")

    def test_effective_sample_size_shrinks_with_the_design_effect(self) -> None:
        sizes = np.full(20, 50.0)
        assert effective_sample_size(1000, sizes, 0.0) == pytest.approx(1000)
        assert effective_sample_size(1000, sizes, 0.1) < 200


class TestCalendarTime:
    def test_portfolio_holds_names_inside_the_window(self) -> None:
        from conftest import SESSIONS, make_panel

        n = len(SESSIONS)
        panel = make_panel({"AAA": [100.0 * 1.01**i for i in range(n)], "SPY": [100.0] * n})
        events = pd.DataFrame(
            [{"ticker": "AAA", "reaction_pos": 10, "category": "Beat", "season": "2020Q1"}]
        )
        portfolio = calendar_time_returns(events, panel, window=5)
        held = portfolio[portfolio["n_long"] > 0]
        assert len(held) == 5
        assert held.index[0] == SESSIONS[11]
        assert held["long_ret"].to_numpy() == pytest.approx(np.full(5, 0.01))

    def test_alpha_recovers_a_planted_daily_return(self) -> None:
        dates = pd.bdate_range("2015-01-01", periods=800)
        rng = np.random.default_rng(2)
        market = rng.normal(0, 0.01, len(dates))
        alpha = 0.0004
        portfolio = pd.DataFrame({"spread_ret": alpha + 0.2 * market}, index=dates)
        factors = pd.DataFrame(
            {
                "mkt_rf": market,
                "smb": rng.normal(0, 0.004, len(dates)),
                "hml": rng.normal(0, 0.004, len(dates)),
                "mom": rng.normal(0, 0.005, len(dates)),
                "rf": 0.00005,
            },
            index=dates,
        )
        result, loadings = calendar_time_alpha(portfolio, factors, hac_lags=5)
        assert result.estimate == pytest.approx(alpha, abs=5e-5)
        assert loadings["mkt_rf"] == pytest.approx(0.2, abs=0.02)
        assert result.p_value < 0.01

    def test_alpha_is_not_found_in_pure_factor_exposure(self) -> None:
        """A book that is only market exposure has no alpha to find.

        The portfolio carries idiosyncratic noise on top of its market
        loading, which is not decoration. An exact multiple of the factor
        gives a regression with no residual variance at all: the fit is
        perfect to machine precision, the standard error collapses to about
        1e-21, and the t-statistic is then a report on the host's floating
        point rather than on the data. An earlier version of this test did
        exactly that, passed on my machine and failed on CI.
        """
        dates = pd.bdate_range("2015-01-01", periods=800)
        rng = np.random.default_rng(9)
        market = rng.normal(0, 0.01, len(dates))
        noise = rng.normal(0, 0.004, len(dates))
        portfolio = pd.DataFrame({"spread_ret": 0.9 * market + noise}, index=dates)
        factors = pd.DataFrame(
            {"mkt_rf": market, "smb": 0.0, "hml": 0.0, "mom": 0.0, "rf": 0.0}, index=dates
        )
        result, loadings = calendar_time_alpha(
            portfolio, factors, factor_columns=("mkt_rf",), hac_lags=5
        )
        # The regression has to be non-degenerate for its p-value to mean
        # anything, so assert that before reading it.
        assert result.std_error > 1e-6
        assert loadings["mkt_rf"] == pytest.approx(0.9, abs=0.05)
        assert result.p_value > 0.05


class TestVectorisedClusterRobust:
    """The fast path must equal the slow one it replaced.

    The cluster-robust variance in :func:`wild_cluster_bootstrap` is computed
    as a quadratic form in one row of the inverse Gram matrix, and the
    per-cluster score sums are accumulated with a scatter-add across every
    bootstrap replication at once. That is several algebraic steps away from
    the textbook sandwich, and a sign or an index slip in it would not be
    visible in any of the behavioural tests above — the p-values would simply
    be wrong by a plausible-looking amount.
    """

    def test_matches_a_textbook_sandwich_estimator(self) -> None:
        rng = np.random.default_rng(5)
        n_obs, n_clusters = 400, 12
        codes = rng.integers(0, n_clusters, n_obs)
        beat = (rng.random(n_obs) > 0.5).astype(float)
        y = 1.5 * beat + rng.normal(0, 3, n_obs) + rng.normal(0, 2, n_clusters)[codes]
        events = pd.DataFrame(
            {
                "value": y,
                "category": np.where(beat > 0, "Beat", "Miss"),
                "season": [f"s{c}" for c in codes],
            }
        )

        result = wild_cluster_bootstrap(events, "value", n_boot=50, seed=1)

        design = np.column_stack([np.ones(n_obs), beat])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        resid = y - design @ beta
        gram_inv = np.linalg.inv(design.T @ design)
        meat = np.zeros((2, 2))
        for cluster in range(n_clusters):
            rows = codes == cluster
            score = design[rows].T @ resid[rows]
            meat += np.outer(score, score)
        adjust = n_clusters / (n_clusters - 1) * (n_obs - 1) / (n_obs - 2)
        se_reference = float(np.sqrt((adjust * gram_inv @ meat @ gram_inv)[1, 1]))

        assert result.estimate == pytest.approx(float(beta[1]), abs=1e-12)
        assert result.std_error == pytest.approx(se_reference, abs=1e-12)
        assert result.statistic == pytest.approx(float(beta[1]) / se_reference, abs=1e-12)
