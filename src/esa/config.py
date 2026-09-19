"""Paths, tunable defaults and run metadata for the PEAD study.

Every default here is an assumption that shows up in a published number, so
each one carries the reason it was chosen rather than a bare literal.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
UNIVERSE_DIR = DATA_DIR / "universe"
SAMPLE_DIR = DATA_DIR / "sample"
CACHE_DIR = DATA_DIR / "cache"

OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
RESULTS_DIR = DOCS_DIR / "results"

#: Pinned point-in-time snapshot of the index membership table. The filename
#: carries the as-of date because the table is not versioned upstream.
UNIVERSE_SNAPSHOT = UNIVERSE_DIR / "sp500_constituents_2026-09-19.csv"

BENCHMARK_TICKER = "SPY"

#: Contact address sent to SEC EDGAR. Their access policy requires a
#: descriptive User-Agent with a working e-mail; requests without one are
#: throttled or refused.
SEC_USER_AGENT = "earnings-surprise-analyzer (research; lyonnfl@gmail.com)"

#: EDGAR asks for no more than 10 requests per second.
SEC_MIN_REQUEST_INTERVAL_S = 0.11

#: Post-announcement horizons in trading days, measured from the close of the
#: first session in which the market could react to the announcement.
RETURN_WINDOWS = (0, 1, 5, 10, 20)

#: Standardised unexpected earnings needs a run of prior year-over-year
#: changes to scale by. Bernard & Thomas (1989) use eight quarters; fewer
#: makes the denominator unstable, more pushes the study start date later.
SUE_HISTORY_QUARTERS = 8

#: SUE cut-offs in standard deviations. One sigma puts about 18% of the
#: sample in each extreme bucket and the rest in the middle, which keeps the
#: tails large enough to test while leaving them genuinely extreme. The
#: threshold is a choice, not a finding, so it is stated rather than tuned.
SUE_BEAT_THRESHOLD = 1.0
SUE_MISS_THRESHOLD = -1.0

#: Analyst-surprise cut-offs in percent, kept only so the SUE result can be
#: compared against the definition the first version of this repo used.
PCT_BEAT_THRESHOLD = 5.0
PCT_MISS_THRESHOLD = -5.0


@dataclass(frozen=True)
class CostAssumptions:
    """Round-trip trading frictions, all in basis points of notional."""

    #: Broker commission per side. Large-cap US equity retail/institutional
    #: commissions sit near zero to 1 bp; 0.5 bp is a mid estimate.
    commission_bps: float = 0.5
    #: Half the quoted bid-ask spread per side. S&P 500 names quote roughly
    #: 1-3 bp wide, so a 1 bp half-spread is the middle of that range.
    half_spread_bps: float = 1.0
    #: Square-root market-impact coefficient, applied as
    #: ``impact_coef_bps * sqrt(participation)``. 10 bp at 1% of daily volume
    #: is a conservative reading of the usual square-root-law calibrations.
    impact_coef_bps: float = 10.0
    #: Fraction of a name's average daily volume the strategy assumes it
    #: takes on entry and on exit.
    participation_rate: float = 0.01
    #: Annualised stock-loan fee on short legs. General-collateral mega-caps
    #: borrow in the 25-50 bp range; 40 bp is deliberately not the cheapest.
    borrow_bps_annual: float = 40.0


@dataclass(frozen=True)
class StudyConfig:
    """Everything that changes a published number."""

    start_date: str = "2009-01-01"
    end_date: str = "2026-09-19"
    #: Events before this date are dropped: SUE needs a warm-up window, and
    #: XBRL coverage is thin in the first year of the mandate.
    event_start_date: str = "2012-01-01"
    hold_days: int = 20
    entry_timing: str = "next_close"
    windows: tuple[int, ...] = RETURN_WINDOWS
    sue_history_quarters: int = SUE_HISTORY_QUARTERS
    beat_threshold: float = SUE_BEAT_THRESHOLD
    miss_threshold: float = SUE_MISS_THRESHOLD
    n_bootstrap: int = 10_000
    random_seed: int = 20240719
    costs: CostAssumptions = field(default_factory=CostAssumptions)

    def with_(self, **kwargs: object) -> StudyConfig:
        """Return a copy with fields replaced, for sweeps and tests."""
        return replace(self, **kwargs)  # type: ignore[arg-type]


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for path in (CACHE_DIR, OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR, UNIVERSE_DIR, SAMPLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def git_commit() -> str:
    """Short SHA of the working tree, or ``"unknown"`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "unknown"
    sha = out.stdout.strip()
    return sha or "unknown"


def write_provenance(
    artifact: Path,
    *,
    command: str,
    data_source: str,
    as_of: str,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write the ``.meta.json`` sidecar that every published artifact carries.

    The sidecar is what makes a number in the README traceable: it records the
    command, the code version and the vintage of the data behind the file.
    """
    meta: dict[str, object] = {
        "artifact": artifact.name,
        "command": command,
        "git_commit": git_commit(),
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": data_source,
        "data_as_of": as_of,
    }
    if extra:
        meta.update(extra)
    sidecar = artifact.with_suffix(artifact.suffix + ".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return sidecar
