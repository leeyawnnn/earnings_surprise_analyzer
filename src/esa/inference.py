"""Inference that takes the dependence in the sample seriously.

Earnings events are not independent observations, in two ways that both push
the same direction.

*Calendar overlap.* A twenty-session window opened by a company reporting on
the third Tuesday of the season overlaps almost entirely with the window of a
company reporting on the fourth. Those two "observations" share most of the
same market shocks.

*Cross-sectional correlation.* Firms in the same index, often in the same
sector, reporting within the same fortnight, have residuals that stay
correlated after the benchmark is subtracted, because a broad-market
benchmark does not span sector or style moves.

A Welch t-test over pooled events assumes neither problem exists. It will
report an effective sample of several thousand when the number of genuinely
independent observations is closer to the number of earnings seasons. This
module reports the naive test — so the gap is visible — alongside three
corrections that each attack the dependence a different way:

* :func:`block_bootstrap_spread` resamples whole seasons, so any correlation
  inside a season is carried along intact.
* :func:`wild_cluster_bootstrap` refits the regression with the null imposed
  and sign-flips residuals season by season.
* :func:`calendar_time_alpha` sidesteps the problem rather than correcting
  for it, by collapsing everything into one daily portfolio return series.

References
----------
Fama (1998), "Market efficiency, long-term returns, and behavioral finance",
JFE 49(3), 283-306, section 4, on why calendar-time portfolios are preferred
for overlapping long-horizon events.
Mitchell & Stafford (2000), "Managerial Decisions and Long-Term Stock Price
Performance", Journal of Business 73(3), 287-329, on cross-sectional
dependence inflating event-study test statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .prices import PricePanel


@dataclass(frozen=True)
class TestResult:
    """One hypothesis test, whatever produced it."""

    label: str
    metric: str
    estimate: float
    std_error: float
    statistic: float
    p_value: float
    n_obs: int
    n_clusters: int
    method: str
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SeasonStats:
    """Per-cluster sums and counts, so a resample costs no dataframe work.

    Every cluster bootstrap in this module resamples whole earnings seasons and
    recomputes the same Beat-minus-Miss mean spread. Rebuilding a dataframe for
    each of ten thousand draws is what makes that approach feel expensive; the
    statistic only needs four numbers per season, so it is not.

    Resampling seasons and taking the ratio of summed totals to summed counts
    reproduces the pooled mean on the resampled data exactly.
    """

    beat_sum: np.ndarray
    beat_n: np.ndarray
    miss_sum: np.ndarray
    miss_n: np.ndarray
    keys: np.ndarray

    @property
    def n_clusters(self) -> int:
        return len(self.keys)

    @classmethod
    def build(cls, events: pd.DataFrame, metric: str, cluster: str = "season") -> SeasonStats:
        usable = events.dropna(subset=[metric])
        codes, keys = pd.factorize(usable[cluster], sort=True)
        values = usable[metric].to_numpy(dtype=float)
        is_beat = (usable["category"] == "Beat").to_numpy()
        is_miss = (usable["category"] == "Miss").to_numpy()
        size = len(keys)
        return cls(
            beat_sum=np.bincount(codes[is_beat], weights=values[is_beat], minlength=size),
            beat_n=np.bincount(codes[is_beat], minlength=size).astype(float),
            miss_sum=np.bincount(codes[is_miss], weights=values[is_miss], minlength=size),
            miss_n=np.bincount(codes[is_miss], minlength=size).astype(float),
            keys=np.asarray(keys),
        )

    def spread(self, picks: np.ndarray) -> float:
        """Beat-minus-Miss mean spread over a multiset of cluster indices."""
        return float(self.spread_many(picks[None, :])[0])

    def spread_many(self, picks: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`spread` over a ``(n_draws, n_clusters)`` index array."""
        beat_sum = self.beat_sum[picks].sum(axis=1)
        beat_n = self.beat_n[picks].sum(axis=1)
        miss_sum = self.miss_sum[picks].sum(axis=1)
        miss_n = self.miss_n[picks].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            beat_mean = np.where(beat_n > 0, beat_sum / np.where(beat_n > 0, beat_n, 1), np.nan)
            miss_mean = np.where(miss_n > 0, miss_sum / np.where(miss_n > 0, miss_n, 1), np.nan)
        return beat_mean - miss_mean


def _split(events: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = events.dropna(subset=[metric])
    return usable[usable["category"] == "Beat"], usable[usable["category"] == "Miss"]


def welch_spread(events: pd.DataFrame, metric: str) -> TestResult:
    """Welch's t-test on the Beat-minus-Miss mean difference.

    Reported for comparison only. Its standard error is the one this module
    exists to argue against.
    """
    beats, misses = _split(events, metric)
    a, b = beats[metric].to_numpy(), misses[metric].to_numpy()
    if len(a) < 2 or len(b) < 2:
        return TestResult("Beat - Miss", metric, np.nan, np.nan, np.nan, np.nan, 0, 0, "welch")
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    diff = float(a.mean() - b.mean())
    return TestResult(
        label="Beat - Miss",
        metric=metric,
        estimate=diff,
        std_error=se,
        statistic=float(t_stat),
        p_value=float(p_val),
        n_obs=len(a) + len(b),
        n_clusters=int(events["season"].nunique()),
        method="welch (assumes independence)",
        ci_low=diff - 1.96 * se,
        ci_high=diff + 1.96 * se,
    )


def block_bootstrap_spread(
    events: pd.DataFrame,
    metric: str,
    *,
    cluster: str = "season",
    n_boot: int = 10_000,
    seed: int = 0,
) -> TestResult:
    """Resample whole earnings seasons with replacement.

    Each draw takes as many seasons as the sample contains, with replacement,
    and keeps every event in a drawn season together. Whatever correlates
    inside a season — shared market moves, sector news, a common macro shock —
    survives resampling untouched, so the spread of the bootstrap distribution
    reflects it.

    The p-value is the recentred two-sided one: the fraction of draws whose
    deviation from the point estimate is at least as large as the point
    estimate itself, which tests the null that the true spread is zero.
    """
    usable = events.dropna(subset=[metric])
    stats_ = SeasonStats.build(usable, metric, cluster)
    if stats_.n_clusters < 2:
        return TestResult(
            "Beat - Miss",
            metric,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0,
            stats_.n_clusters,
            "block bootstrap",
        )

    point = stats_.spread(np.arange(stats_.n_clusters))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, stats_.n_clusters, size=(n_boot, stats_.n_clusters))
    draws = stats_.spread_many(picks)

    draws = draws[np.isfinite(draws)]
    se = float(draws.std(ddof=1))
    p_val = float(np.mean(np.abs(draws - point) >= abs(point)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return TestResult(
        label="Beat - Miss",
        metric=metric,
        estimate=point,
        std_error=se,
        statistic=point / se if se > 0 else np.nan,
        p_value=max(p_val, 1.0 / (len(draws) + 1)),
        # Count only the events the statistic uses, so n_obs is comparable
        # with the other tests in the same table.
        n_obs=int((usable["category"].isin(["Beat", "Miss"])).sum()),
        n_clusters=stats_.n_clusters,
        method=f"block bootstrap by {cluster}, {len(draws)} draws",
        ci_low=float(lo),
        ci_high=float(hi),
    )


def wild_cluster_bootstrap(
    events: pd.DataFrame,
    metric: str,
    *,
    cluster: str = "season",
    n_boot: int = 10_000,
    seed: int = 0,
) -> TestResult:
    """Wild cluster bootstrap-t on a Beat dummy, null imposed.

    The regression is ``y = a + b * beat + e`` over Beat and Miss events only,
    so ``b`` is the same Beat-minus-Miss spread the other tests estimate. Each
    replication refits under ``b = 0``, multiplies the restricted residuals by
    a Rademacher draw that is constant within a season, and re-estimates. The
    reference distribution for the cluster-robust t-statistic is the
    bootstrap's own, not the normal.

    This matters because cluster-robust standard errors are badly behaved when
    clusters are few: with a handful of seasons they are severely
    downward-biased, and the wild cluster bootstrap is the standard remedy
    (Cameron, Gelbach & Miller 2008). It stays the safer choice even with the
    fifty-odd seasons this sample has.
    """
    usable = events.dropna(subset=[metric])
    usable = usable[usable["category"].isin(["Beat", "Miss"])]
    if usable.empty:
        return TestResult(
            "Beat - Miss", metric, np.nan, np.nan, np.nan, np.nan, 0, 0, "wild cluster bootstrap"
        )

    y = usable[metric].to_numpy(dtype=float)
    beat = (usable["category"] == "Beat").to_numpy(dtype=float)
    x = np.column_stack([np.ones_like(beat), beat])
    codes, keys = pd.factorize(usable[cluster])
    n_clusters = len(keys)
    if n_clusters < 2:
        return TestResult(
            "Beat - Miss",
            metric,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            len(y),
            n_clusters,
            "wild cluster bootstrap",
        )

    xtx_inv = np.linalg.inv(x.T @ x)
    projector = xtx_inv @ x.T
    # Small-cluster correction, as in Cameron & Miller (2015) eq. 12.
    adjust = n_clusters / max(n_clusters - 1, 1) * (len(y) - 1) / max(len(y) - 2, 1)
    # cov = adjust * A @ meat @ A, and only the slope variance is needed, which
    # reduces to a quadratic form in the second row of A.
    b, c = float(xtx_inv[1, 0]), float(xtx_inv[1, 1])

    def slope_and_t(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Slope and cluster-robust t for each column of ``targets``."""
        beta = projector @ targets
        resid = targets - x @ beta
        s0 = np.zeros((n_clusters, targets.shape[1]))
        s1 = np.zeros((n_clusters, targets.shape[1]))
        np.add.at(s0, codes, resid)
        np.add.at(s1, codes, resid * beat[:, None])
        variance = adjust * (
            b**2 * (s0**2).sum(axis=0)
            + 2 * b * c * (s0 * s1).sum(axis=0)
            + c**2 * (s1**2).sum(axis=0)
        )
        se = np.sqrt(np.maximum(variance, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            return beta[1], np.where(se > 0, beta[1] / se, np.nan)

    slope, t_obs = slope_and_t(y[:, None])
    point = float(slope[0])
    t_point = float(t_obs[0])
    se_point = abs(point / t_point) if t_point not in (0.0,) and np.isfinite(t_point) else np.nan

    # Restricted fit: drop the Beat dummy so the null b = 0 holds by construction.
    x0 = x[:, :1]
    fitted0 = x0 @ np.linalg.lstsq(x0, y, rcond=None)[0]
    resid0 = y - fitted0

    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    chunk = max(1, min(n_boot, 2_000_000 // max(len(y), 1)))
    remaining = n_boot
    while remaining > 0:
        size = min(chunk, remaining)
        weights = rng.choice(np.array([-1.0, 1.0]), size=(n_clusters, size))[codes]
        draws.append(slope_and_t(fitted0[:, None] + resid0[:, None] * weights)[1])
        remaining -= size

    t_star = np.concatenate(draws)
    t_star = t_star[np.isfinite(t_star)]
    p_val = float(np.mean(np.abs(t_star) >= abs(t_point))) if len(t_star) else np.nan
    return TestResult(
        label="Beat - Miss",
        metric=metric,
        estimate=point,
        std_error=se_point,
        statistic=t_point,
        p_value=max(p_val, 1.0 / (len(t_star) + 1)),
        n_obs=len(y),
        n_clusters=n_clusters,
        method=f"wild cluster bootstrap-t by {cluster}, {len(t_star)} draws",
        ci_low=point - 1.96 * se_point,
        ci_high=point + 1.96 * se_point,
    )


# ── calendar-time portfolio ──────────────────────────────────────────────


def calendar_time_returns(
    events: pd.DataFrame,
    panel: PricePanel,
    *,
    window: int = 20,
    long_bucket: str = "Beat",
    short_bucket: str = "Miss",
) -> pd.DataFrame:
    """Daily returns of a portfolio that holds every name inside its window.

    On each session the book holds, equal-weighted, every company whose
    reaction session was within the last ``window`` sessions. Overlap stops
    being a statistical nuisance because it is now just a position size: two
    simultaneous events share the day's capital rather than contributing two
    correlated "observations".

    Returns a frame indexed by session with the two legs, the long-short
    spread, and how many names sat in each leg.
    """
    closes = panel.close
    daily = closes.pct_change().to_numpy()
    n_sessions, _ = daily.shape

    codes, index = pd.factorize(events["ticker"])
    col_of = {t: i for i, t in enumerate(closes.columns)}
    col = np.array([col_of.get(t, -1) for t in index])[codes]
    keep = col >= 0
    rpos = events["reaction_pos"].to_numpy()[keep]
    bucket = events["category"].to_numpy()[keep]
    col = col[keep]

    offsets = np.arange(1, window + 1)
    rows = (rpos[:, None] + offsets[None, :]).ravel()
    cols = np.repeat(col, window)
    buckets = np.repeat(bucket, window)
    inside = rows < n_sessions
    rows, cols, buckets = rows[inside], cols[inside], buckets[inside]

    values = daily[rows, cols]
    finite = np.isfinite(values)
    rows, cols, buckets, values = rows[finite], cols[finite], buckets[finite], values[finite]

    out = pd.DataFrame(index=closes.index)
    for name, label in (("long", long_bucket), ("short", short_bucket)):
        mask = buckets == label
        total = np.zeros(n_sessions)
        count = np.zeros(n_sessions)
        np.add.at(total, rows[mask], values[mask])
        np.add.at(count, rows[mask], 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{name}_ret"] = np.where(count > 0, total / np.where(count > 0, count, 1), np.nan)
        out[f"n_{name}"] = count.astype(int)

    out["spread_ret"] = out["long_ret"].fillna(0.0) - out["short_ret"].fillna(0.0)
    out.loc[(out["n_long"] == 0) & (out["n_short"] == 0), "spread_ret"] = np.nan
    return out


def calendar_time_alpha(
    portfolio: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    column: str = "spread_ret",
    factor_columns: tuple[str, ...] = ("mkt_rf", "smb", "hml", "mom"),
    excess: bool = False,
    hac_lags: int | None = None,
) -> tuple[TestResult, pd.Series]:
    """Regress the daily portfolio return on factors and test alpha = 0.

    Standard errors are Newey-West. Even after the calendar-time collapse the
    residuals carry moving-average structure of roughly the holding period,
    because the same names sit in the book on consecutive days, so the lag
    length defaults to the holding window.

    ``excess`` subtracts the risk-free rate, which is right for a long-only
    leg and wrong for the zero-investment long-short spread.
    """
    import statsmodels.api as sm

    joined = portfolio[[column]].join(factors, how="inner").dropna()
    if joined.empty:
        empty = TestResult("alpha", column, np.nan, np.nan, np.nan, np.nan, 0, 0, "calendar-time")
        return empty, pd.Series(dtype=float)

    y = joined[column].to_numpy()
    if excess:
        y = y - joined["rf"].to_numpy()
    used = [c for c in factor_columns if c in joined.columns]
    design = sm.add_constant(joined[used].to_numpy())
    lags = hac_lags if hac_lags is not None else 20
    model = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    loadings = pd.Series(model.params[1:], index=used)
    alpha_daily = float(model.params[0])
    result = TestResult(
        label="alpha (daily)",
        metric=column,
        estimate=alpha_daily,
        std_error=float(model.bse[0]),
        statistic=float(model.tvalues[0]),
        p_value=float(model.pvalues[0]),
        n_obs=len(joined),
        n_clusters=len(joined),
        method=f"calendar-time OLS on {'+'.join(used)}, Newey-West {lags} lags",
        ci_low=float(model.conf_int()[0][0]),
        ci_high=float(model.conf_int()[0][1]),
    )
    return result, loadings


# ── how much sample would be enough ──────────────────────────────────────


def effective_sample_size(n_obs: int, cluster_sizes: np.ndarray, icc: float) -> float:
    """Sample size a clustered design is worth, under a design effect.

    The design effect for clusters of average size ``m`` and intra-cluster
    correlation ``rho`` is ``1 + (m - 1) * rho``; dividing the raw count by it
    gives the number of independent observations the design is equivalent to.
    """
    if n_obs == 0 or len(cluster_sizes) == 0:
        return 0.0
    mean_size = float(np.mean(cluster_sizes))
    design_effect = 1.0 + (mean_size - 1.0) * icc
    return n_obs / max(design_effect, 1e-9)


def intra_cluster_correlation(events: pd.DataFrame, metric: str, cluster: str = "season") -> float:
    """One-way random-effects estimate of the intra-cluster correlation.

    ``rho = var_between / (var_between + var_within)`` from a one-way ANOVA
    decomposition. This is the number that decides how badly the naive test
    overstates its precision.
    """
    usable = events.dropna(subset=[metric])
    grouped = usable.groupby(cluster)[metric]
    sizes = grouped.size().to_numpy(dtype=float)
    if len(sizes) < 2:
        return 0.0
    means = grouped.mean().to_numpy()
    grand = float(usable[metric].mean())
    k = len(sizes)
    n = float(sizes.sum())

    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(
        np.sum((usable[metric].to_numpy() - usable[cluster].map(grouped.mean()).to_numpy()) ** 2)
    )
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / max(n - k, 1)
    m0 = (n - float(np.sum(sizes**2)) / n) / (k - 1)
    var_between = max((ms_between - ms_within) / max(m0, 1e-9), 0.0)
    total = var_between + ms_within
    return float(var_between / total) if total > 0 else 0.0


def required_sample_size(
    effect: float,
    sd: float,
    *,
    icc: float = 0.0,
    cluster_size: float = 1.0,
    variance_inflation: float | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Events needed for ``power`` against a two-sided test, given clustering.

    The usual two-sample formula, inflated for dependence. The inflation can
    be given directly as ``variance_inflation`` — the ratio of clustered to
    naive sampling variance — or derived from an intra-cluster correlation
    through the design effect ``1 + (m - 1) * rho``.

    Prefer the measured inflation. The design effect describes clustering in
    the *level* of returns, and a Beat-minus-Miss spread differences the
    common component out, so the design effect overstates how much precision
    is actually lost. This repo reports both and uses the measured one.
    """
    if effect == 0 or sd <= 0:
        return float("inf")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    per_arm = 2 * (z_alpha + z_beta) ** 2 * sd**2 / effect**2
    if variance_inflation is None:
        variance_inflation = 1.0 + (cluster_size - 1.0) * icc
    return float(2 * per_arm * variance_inflation)


def achieved_power(effect: float, std_error: float, *, alpha: float = 0.05) -> float:
    """Probability a repeat of this design would reject, if the effect is real."""
    if not np.isfinite(effect) or not np.isfinite(std_error) or std_error <= 0:
        return float("nan")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(abs(effect) / std_error - z_alpha))


def results_frame(results: list[TestResult]) -> pd.DataFrame:
    """Collect test results into the table the README quotes."""
    return pd.DataFrame([r.as_dict() for r in results])
