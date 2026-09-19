"""SEC EDGAR client: announcement timestamps and first-reported EPS.

Two things come from EDGAR, and both are point-in-time in a way that a
consensus-estimate feed is not.

**When the news hit.** A quarterly result is released in an 8-K carrying item
2.02, "Results of Operations and Financial Condition". EDGAR records the
acceptance timestamp of that filing to the second, so the release time — not
just the date — is observable. That is what decides whether the first tradeable
session is the same day or the next one.

**What was reported.** ``companyconcept`` returns every value a company has
ever tagged for a concept, each stamped with the accession that carried it.
Taking the *earliest* filing for a given fiscal period recovers the number as
first reported, before any later restatement, which is what a trader would
have seen.

What EDGAR does not provide is analyst consensus. That is the reason this
study measures surprise as standardised unexpected earnings rather than as a
deviation from estimates; see :mod:`esa.surprise`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from . import config

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
COMPANY_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"

#: 8-K item number for "Results of Operations and Financial Condition".
EARNINGS_ITEM = "2.02"

#: Diluted EPS is the headline figure and the one Bernard & Thomas-style
#: studies use. A minority of filers tag only basic EPS, so it is the
#: fallback rather than a parallel series.
EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")

#: A quarterly XBRL duration is nominally 91 days but ranges over 13-week and
#: 3-month conventions, plus 52/53-week retail calendars.
QUARTER_DAYS = (80, 100)
#: An annual duration on the same conventions.
YEAR_DAYS = (350, 380)


@dataclass
class SecClient:
    """Rate-limited EDGAR reader with an on-disk cache.

    EDGAR asks for a descriptive User-Agent and no more than ten requests a
    second. Responses are cached as raw JSON so that re-running the study does
    not re-hit the API and so that a cached run is byte-reproducible.
    """

    cache_dir: Path = config.CACHE_DIR / "sec"
    user_agent: str = config.SEC_USER_AGENT
    min_interval_s: float = config.SEC_MIN_REQUEST_INTERVAL_S
    timeout_s: int = 45
    max_retries: int = 3
    _last_request: float = 0.0

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_request = time.monotonic()

    def get_json(self, url: str, cache_key: str, *, use_cache: bool = True) -> dict[str, Any] | None:
        """Fetch and cache a JSON document. ``None`` means EDGAR has no such document."""
        path = self.cache_dir / f"{cache_key}.json"
        if use_cache and path.exists():
            text = path.read_text()
            return None if text == "null" else json.loads(text)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = self._session.get(url, timeout=self.timeout_s)
            except requests.RequestException as exc:  # pragma: no cover - network flake
                last_error = exc
                time.sleep(1.0 + attempt)
                continue
            if response.status_code == 404:
                path.write_text("null")
                return None
            if response.status_code in (403, 429, 500, 502, 503):  # pragma: no cover
                last_error = requests.HTTPError(f"{response.status_code} for {url}")
                time.sleep(2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            path.write_text(response.text)
            return response.json()

        raise RuntimeError(f"EDGAR request failed after {self.max_retries} attempts: {url}") from last_error

    # ── announcements ────────────────────────────────────────────────────

    def submissions(self, cik: int, *, use_cache: bool = True) -> pd.DataFrame:
        """Every filing EDGAR lists for a CIK, across all history pages.

        The ``recent`` block holds only the latest thousand filings; older ones
        sit in companion documents named in ``filings.files``. Ignoring those
        silently truncates history around 2015 for an active filer.
        """
        root = self.get_json(
            SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik:010d}", use_cache=use_cache
        )
        if root is None:
            return pd.DataFrame()

        blocks = [root["filings"]["recent"]]
        for page in root["filings"].get("files", []):
            extra = self.get_json(
                SUBMISSIONS_PAGE_URL.format(name=page["name"]),
                f"submissions_{cik:010d}_{page['name'].rsplit('-', 1)[-1].removesuffix('.json')}",
                use_cache=use_cache,
            )
            if extra:
                blocks.append(extra)

        frames = [pd.DataFrame(block) for block in blocks if block]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def earnings_announcements(self, cik: int, *, use_cache: bool = True) -> pd.DataFrame:
        """8-K item 2.02 filings for a CIK, with acceptance timestamps.

        Returns columns ``[cik, accession, filing_date, accepted_utc]``.
        """
        filings = self.submissions(cik, use_cache=use_cache)
        if filings.empty or "items" not in filings.columns:
            return pd.DataFrame(columns=["cik", "accession", "filing_date", "accepted_utc"])

        items = filings["items"].fillna("")
        is_earnings_8k = (filings["form"] == "8-K") & items.str.contains(EARNINGS_ITEM, regex=False)
        hits = filings.loc[is_earnings_8k]
        if hits.empty:
            return pd.DataFrame(columns=["cik", "accession", "filing_date", "accepted_utc"])

        out = pd.DataFrame(
            {
                "cik": cik,
                "accession": hits["accessionNumber"].to_numpy(),
                "filing_date": pd.to_datetime(hits["filingDate"]).to_numpy(),
                "accepted_utc": pd.to_datetime(hits["acceptanceDateTime"], utc=True, format="ISO8601").to_numpy(),
            }
        )
        return out.drop_duplicates("accession").sort_values("accepted_utc").reset_index(drop=True)

    # ── reported EPS ─────────────────────────────────────────────────────

    def eps_facts(self, cik: int, *, use_cache: bool = True) -> pd.DataFrame:
        """Raw EPS facts for a CIK, one row per (period, filing).

        Columns ``[cik, tag, start, end, val, filed, form, fy, fp, days]``.
        """
        for tag in EPS_TAGS:
            payload = self.get_json(
                COMPANY_CONCEPT_URL.format(cik=cik, tag=tag),
                f"eps_{cik:010d}_{tag}",
                use_cache=use_cache,
            )
            if not payload:
                continue
            units = payload.get("units", {}).get("USD/shares")
            if not units:
                continue
            frame = pd.DataFrame(units)
            if frame.empty or "start" not in frame.columns:
                continue
            frame = frame.dropna(subset=["start", "end", "val"])
            frame["cik"] = cik
            frame["tag"] = tag
            frame["start"] = pd.to_datetime(frame["start"])
            frame["end"] = pd.to_datetime(frame["end"])
            frame["filed"] = pd.to_datetime(frame["filed"])
            frame["days"] = (frame["end"] - frame["start"]).dt.days
            keep = ["cik", "tag", "start", "end", "val", "filed", "form", "fy", "fp", "days"]
            return frame[[c for c in keep if c in frame.columns]]
        return pd.DataFrame()


def first_reported(facts: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Collapse raw facts to one first-reported value per fiscal period.

    A company re-tags the same quarter in later filings — in comparative
    columns, and again after a restatement. Sorting by filing date and keeping
    the first occurrence recovers the number as it was originally published,
    which is the only version a trader could have acted on.
    """
    if facts.empty:
        return pd.DataFrame(columns=["cik", "start", "end", "val", "filed", "days"])
    window = facts[(facts["days"] >= lo) & (facts["days"] <= hi)]
    if window.empty:
        return pd.DataFrame(columns=["cik", "start", "end", "val", "filed", "days"])
    return (
        window.sort_values(["end", "filed"])
        .groupby("end", as_index=False)
        .first()
        .loc[:, ["cik", "start", "end", "val", "filed", "days"]]
    )


def quarterly_eps(facts: pd.DataFrame) -> pd.DataFrame:
    """First-reported quarterly EPS, with fiscal Q4 derived from the annual figure.

    Filers report Q1-Q3 EPS in 10-Qs but disclose only the full year in the
    10-K, so the fourth fiscal quarter is missing from XBRL for most
    companies. It is recovered as the annual figure less the three quarters
    that fall inside the same annual period.

    The subtraction is approximate: per-share figures do not add up exactly
    when the diluted share count moves during the year, so derived rows are
    flagged with ``derived=True`` and the study reports its headline result
    both with and without them.
    """
    direct = first_reported(facts, *QUARTER_DAYS)
    if not direct.empty:
        direct = direct.assign(derived=False)

    annual = first_reported(facts, *YEAR_DAYS)
    derived_rows = []
    if not annual.empty and not direct.empty:
        for _, year in annual.iterrows():
            inside = direct[(direct["start"] >= year["start"]) & (direct["end"] <= year["end"])]
            if len(inside) != 3:
                continue
            q4_start = inside["end"].max() + pd.Timedelta(days=1)
            derived_rows.append(
                {
                    "cik": year["cik"],
                    "start": q4_start,
                    "end": year["end"],
                    "val": float(year["val"]) - float(inside["val"].sum()),
                    # The value becomes public in the annual filing, so that is
                    # the date at which it is known.
                    "filed": year["filed"],
                    "days": (year["end"] - q4_start).days,
                    "derived": True,
                }
            )

    parts = [f for f in (direct, pd.DataFrame(derived_rows)) if not f.empty]
    if not parts:
        return pd.DataFrame(columns=["cik", "start", "end", "val", "filed", "days", "derived"])

    out = pd.concat(parts, ignore_index=True)
    # A derived quarter must never displace a directly reported one.
    out = out.sort_values(["end", "derived"]).drop_duplicates("end", keep="first")
    return out.sort_values("end").reset_index(drop=True)
