# What was broken, what changed, and what I retracted

Temporary. This is for you to read, fold anything still useful into the README's Limitations
section, and delete.

---

## 1. The defect that invalidated every published number

**The study dated each event by the fiscal period end, not by the announcement.**

`data_fetcher.py` tried `Ticker.get_earnings_dates()` first, whose index is the announcement
datetime, and fell back to `Ticker.earnings_history`, whose index is the **fiscal quarter end**. The
fallback is what actually ran: all 100 rows in the committed `earnings_raw.csv` had month-end dates
— 2025-04-30, 2025-05-31, 2025-06-30 and so on — and no company announces on the last calendar day
of its own quarter.

Checked against EDGAR for the quarter ending 2025-04-30:

| Company | Period end | 8-K item 2.02 accepted | Lag |
|---|---|---|---:|
| CRM | 2025-04-30 | 2025-05-28 16:05 ET | 28 days |
| NVDA | 2025-04-30 | 2025-05-28 16:21 ET | 28 days |
| WMT | 2025-04-30 | 2025-05-15 06:58 ET | 15 days |
| AAPL | 2025-06-30 | 2025-07-31 16:30 ET | 31 days |

So the "Day 0" return was measured two to four weeks before the result was public, and the
"Day 1–20 drift window" *contained* the announcement it claimed to be drifting after. The strategy
opened positions on dates when the surprise it was trading on did not yet exist.

This is worth being blunt about, because the brief I was working from said the backtest logic was
sound and had no look-ahead in the entry timing. The entry timing was sound relative to the event
date; the event date was wrong by a month. The +13.54% Day 1–20 spread at p = 0.006 was in
substantial part the announcement jump itself, captured by a position taken on the strength of
information not yet public.

**Fixed by** matching each fiscal quarter to the first 8-K carrying item 2.02 accepted after the
period closes, using EDGAR's acceptance timestamp. `tests/test_pipeline.py` pins it.

---

## 2. Retracted claims

Every number below appeared in the old README. None survived.

| Old claim | Status | Replacement |
|---|---|---|
| Day 1–20 Beat–Miss spread **+13.54%**, p = 0.006 | Retracted | **+2.12 pp** day 0–20, p < 0.0001; **+0.78 pp** drift only, p = 0.0019 corrected |
| Day 1–10 spread +5.19%, p = 0.033 | Retracted | +0.47 pp drift only, p = 0.010 corrected |
| Day 0 spread +1.14%, p = 0.041 | Retracted | +1.32 pp, p < 0.0001 |
| Correlation surprise vs return rises to **r = 0.50** at Day 20 | Retracted, not replaced | R² = 0.006 against the day-0 reaction. The old figure was computed against a surprise variable that was partly mis-scaled (see §3) over a window containing the announcement |
| Market-adjusted Day-20 spread +12.3%, p = 0.009 | Retracted | +2.12 pp |
| Sharpe ratio **−0.31** | Retracted | **+0.05** entering at the next close, +0.28 at the next open. The old figure was not a Sharpe ratio: it annualised the dispersion of 56 overlapping trade returns |
| Max drawdown −15.82% | Retracted | −26.6%, on a daily-marked equity curve, October 2018 to February 2020 |
| Total return −4.45% vs SPY +11.85% | Retracted | +27.1% vs SPY +657.9% over 14.7 years |
| "No transaction costs — slippage, commissions and short-borrow fees are ignored" | Fixed | Costed. Break-even round trip is **11 bp** entering at the close, 20 bp at the open |
| "~100 events over one year" | Fixed | 17,724 events, 444 companies, 14.7 years |

The direction is what you would expect once the look-ahead is removed: the effect is roughly six
times smaller than claimed, and the strategy's apparent failure was also wrong — it loses to the
index by much more than the old version said.

---

## 3. Additional defects found, not in the brief

**`Surprise(%)` was multiplied by a hundred.** `data_fetcher.py` treated Yahoo's `Surprise(%)`
column as a fraction and scaled it, then capped the result at ±200. The field is already a
percentage: Apple's 2026-07-30 quarter reports 6.74, meaning 6.74%. Any row using it became exactly
+200 or −200. In the committed data only three rows were affected, because the fallback path
supplied `NaN` for that column and the code recomputed from EPS — so the bug was mostly masked by
another bug. It would have corrupted the whole sample the moment the primary path succeeded.

**`requirements.txt` pinned a yfinance that no longer works.** `yfinance>=0.2.30` resolves to a
build Yahoo now rejects: `yf.download` returns an empty frame and logs "No timezone found, symbol
may be delisted" for every symbol including SPY. The documented quickstart could not reproduce the
published results on a clean install. Now `yfinance>=1.0`, with exact versions in
`requirements.lock.txt`.

**A few filers mis-tag `EarningsPerShareDiluted`.** Intercontinental Exchange tags 120,000,000 for a
2016 quarter — whole-dollar income, not per share. Berkshire tags the Class A figure while the index
member is the Class B, three orders of magnitude apart. Unfiltered, these produced SUE values above
a million. Filtered by a price-relative plausibility test rather than a flat dollar cap, since NVR
genuinely earns over a hundred dollars a share in a quarter.

**Exit-day turnover was never charged.** My own bug, caught by a hand-computed test: when the book
emptied completely the loop skipped the session where the unwind happens, so a round trip cost about
half what it should.

**Per-trade returns were summed, not compounded.** Also mine, also caught by a hand-computed test.
A four-session trade at 1% a day was reported as 4.00% rather than 4.06%.

**`PricePanel` carried a volume frame nothing read.** It was the largest file in the committed
sample. Removed; the cost model charges impact at an assumed participation rate, so storing realised
volume implied a precision the model does not have.

---

## 4. Where I did something other than the brief asked

**Risk-free rate.** The brief suggested FRED `DGS3MO`. I used the one-month Treasury bill return
from the Ken French daily file instead. It is already expressed as a daily simple rate, which is
what a Sharpe calculation needs, and it comes from a source the pipeline already downloads for the
factor model — so it removes a dependency rather than adding one. Stated in the README's data table.

**Cluster-robust standard errors with few clusters.** The brief anticipated four clusters, since the
original sample was four quarters, and asked for the wild cluster bootstrap on that basis. Expanding
the period gives 59 earnings seasons, which is comfortably enough for ordinary cluster-robust
inference. I implemented the wild cluster bootstrap anyway — it is the safer choice and costs
nothing — but the argument for it here is different from the one in the brief.

**Figure resolution.** The figure standard asks for PNG at 200 DPI and separately for a maximum
rendered width of 1400 px. On a 12-inch canvas those conflict: 200 DPI gives 2400 px. I kept the
width cap, since that is what governs how the figure actually renders on GitHub, and emit the one
raster figure at 116 DPI for 1392 px.

**The decay figure did not show decay.** The brief suggested that a decay analysis would likely be
the headline result and that PEAD has probably faded since it became well known. The data does not
support that: the rolling three-year spread strengthened into 2022 and fell back afterwards, with a
mildly positive linear trend. I published the figure and wrote the caption to what it shows.

**Power turned out fine, which was not the expectation.** The brief expected the study to be
badly underpowered and suggested reporting that as a finding. After expansion the sample has 86%
power against the observed effect. The finding worth reporting is the other one: the *original*
100-event sample had about 6% power, so it could not have detected this effect and whatever it did
detect was not this.

---

## 5. Done since, for the record

The CI badge is in the README: the workflow ran green on `main` across seven jobs — lint and
types, tests on 3.11/3.12/3.13, the full sample pipeline, and the quickstart on Linux and macOS.

Getting it green took one fix worth knowing about. `test_alpha_is_not_found_in_pure_factor_exposure`
built its portfolio as an exact multiple of the market factor, which leaves a regression with no
residual variance: the standard error collapsed to about 1e-21 and the t-statistic was reporting on
the runner's linear algebra library rather than on the data. It passed on my machine and failed on
CI, which is the only useful thing it ever did. The portfolio now carries idiosyncratic noise and
the test asserts the standard error is non-degenerate before reading the p-value.

The repository description and topics are set:

```
Event study of post-earnings announcement drift on US equities: surprise measurement, abnormal
return windows, corrected inference, and a costed backtest.

event-study  pead  earnings  quantitative-finance  backtesting  python  equity-research
```

**Now delete this file.**
