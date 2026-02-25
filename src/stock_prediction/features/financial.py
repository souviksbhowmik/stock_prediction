"""Financial report features with point-in-time correct merging.

Features are derived from quarterly income statement, balance sheet, and cash
flow data fetched via yfinance.  Three categories are produced:

Fundamental ratios (prefix ``fin_``)
    Revenue/earnings growth QoQ and YoY, profit margins, leverage, return
    on equity, cash-flow quality, EPS surprise, etc.

Report aging features
    report_age_days    — calendar days since the most recent report was
                         published (0 on announcement day, grows daily).
    report_effect      — exp(−λ × report_age_days), λ = ln(2)/30.
                         Starts at 1.0, halves every 30 days, approaching
                         zero after ~3 months.  Captures diminishing market
                         relevance of stale data.
    report_freshness   — max(0, 91 − report_age_days) / 91.
                         Linear decay from 1.0 on publish day to 0.0 when
                         the next quarterly report is expected.
    days_to_next_report — max(0, 91 − report_age_days).

Point-in-time correctness
    yfinance returns *fiscal quarter-end* dates (e.g. 2023-09-30).  Actual
    earnings releases for large-cap Indian stocks arrive 30–60 days after
    quarter close.  We apply ``announcement_lag_days`` (default 45) to
    convert quarter-end → assumed announcement date, preventing look-ahead
    bias.  When ``ticker.earnings_dates`` is available the actual date is
    used instead.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from stock_prediction.utils.logging import get_logger

logger = get_logger("features.financial")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HALF_LIFE_DAYS: float = 30.0
_LAMBDA: float = math.log(2) / _HALF_LIFE_DAYS    # ≈ 0.0231 per day
_QUARTER_DAYS: int = 91                            # approx calendar days / quarter
_DEFAULT_LAG: int = 45                             # fallback announcement lag (days)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FinancialFeatureGenerator:
    """Fetch quarterly financial reports and engineer point-in-time features.

    Usage
    -----
    gen = FinancialFeatureGenerator()
    df  = gen.merge_financial_features(ohlcv_df, "RELIANCE.NS")
    """

    def __init__(self, announcement_lag_days: int = _DEFAULT_LAG):
        self.lag = announcement_lag_days
        self._cache: dict[str, pd.DataFrame | None] = {}

    # ── Public ────────────────────────────────────────────────────────────

    def merge_financial_features(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Merge quarterly financial features onto an OHLCV/indicator DataFrame.

        For each row in ``df`` (indexed by date), the most recent report whose
        announcement date is ≤ that date is looked up and its ratios are
        attached.  Aging columns are then computed from the per-row age.
        Rows before any known report date get NaN for all financial columns.

        Parameters
        ----------
        df     : DataFrame with a DatetimeIndex (daily OHLCV + indicators).
        symbol : NSE/BSE ticker, e.g. ``"RELIANCE.NS"``.

        Returns
        -------
        df with ``fin_*`` and aging columns added in-place.
        """
        report_df = self._get_report_df(symbol)
        if report_df is None or report_df.empty:
            logger.warning(f"No financial report data for {symbol} — skipping")
            return df

        # Ensure both indices are timezone-naive for merge
        ohlcv_idx = _strip_tz_index(df.index)
        rep_idx   = _strip_tz_index(report_df.index)
        report_df = report_df.copy()
        report_df.index = rep_idx

        # pd.merge_asof: for each OHLCV date, find the latest report date ≤ it
        left = pd.DataFrame({"__date": ohlcv_idx}, index=ohlcv_idx)
        right = (
            report_df
            .sort_index()
            .rename_axis("__rep_date")
            .reset_index()
        )

        merged = pd.merge_asof(
            left.sort_index(),
            right,
            left_on="__date",
            right_on="__rep_date",
            direction="backward",
        )
        merged.index = left.index   # restore index order after sort

        # Compute aging columns
        if "__rep_date" in merged.columns:
            rep_dates  = pd.to_datetime(merged["__rep_date"])
            curr_dates = pd.to_datetime(merged["__date"])
            age = (curr_dates - rep_dates).dt.days.astype(float)
            age[age < 0] = np.nan   # guard: shouldn't happen with backward merge

            merged["report_age_days"]      = age
            merged["report_effect"]        = np.exp(-_LAMBDA * age)
            merged["report_freshness"]     = np.clip(
                (_QUARTER_DAYS - age) / _QUARTER_DAYS, 0.0, 1.0
            )
            merged["days_to_next_report"]  = np.clip(_QUARTER_DAYS - age, 0.0, None)
            merged = merged.drop(columns=["__rep_date"])

        merged = merged.drop(columns=["__date"]).reindex(ohlcv_idx)

        # Attach to original df
        new_cols = [c for c in merged.columns if c not in df.columns]
        for col in new_cols:
            df[col] = merged[col].values

        logger.info(
            f"Financial features for {symbol}: "
            f"{len(new_cols)} columns added "
            f"({sum(c.startswith('fin_') for c in new_cols)} ratios + "
            f"{sum(not c.startswith('fin_') for c in new_cols)} aging)"
        )
        return df

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_report_df(self, symbol: str) -> pd.DataFrame | None:
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_and_compute(symbol)
        return self._cache[symbol]

    def _fetch_and_compute(self, symbol: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return None

        try:
            ticker = yf.Ticker(symbol)
            fin  = _safe_fetch(ticker, "quarterly_financials")
            bs   = _safe_fetch(ticker, "quarterly_balance_sheet")
            cf   = _safe_fetch(ticker, "quarterly_cashflow")
            ed   = _safe_fetch(ticker, "earnings_dates")

            if fin is None or fin.empty:
                logger.warning(f"No quarterly financials for {symbol}")
                return None

            # Transpose: rows = quarter-end dates, columns = metrics
            fin_t = fin.T.sort_index()
            bs_t  = bs.T.sort_index()  if (bs  is not None and not bs.empty)  else pd.DataFrame()
            cf_t  = cf.T.sort_index()  if (cf  is not None and not cf.empty)  else pd.DataFrame()

            # Map quarter-end dates → announcement dates
            ann_map = _build_announcement_map(fin_t.index, ed, self.lag)
            ann_dates = [ann_map.get(d, d + pd.Timedelta(days=self.lag)) for d in fin_t.index]
            fin_t.index = pd.DatetimeIndex(ann_dates).tz_localize(None)
            fin_t = fin_t.sort_index()

            # Merge balance sheet and cash flow onto the same announcement dates
            for extra in [bs_t, cf_t]:
                if extra.empty:
                    continue
                extra_ann = extra.copy()
                extra_ann.index = pd.DatetimeIndex([
                    ann_map.get(d, d + pd.Timedelta(days=self.lag))
                    for d in extra_ann.index
                ]).tz_localize(None)
                extra_ann = extra_ann.sort_index()
                fin_t = fin_t.join(extra_ann, how="left", rsuffix="_dup")
                # Drop duplicate columns from rsuffix collision
                dup_cols = [c for c in fin_t.columns if c.endswith("_dup")]
                fin_t = fin_t.drop(columns=dup_cols)

            # EPS surprise from earnings_dates
            eps_df = _extract_eps_surprise(ed)
            if eps_df is not None:
                fin_t = fin_t.join(eps_df, how="left")

            # Compute derived ratios
            ratio_df = self._compute_ratios(fin_t)
            if ratio_df.empty:
                logger.warning(f"No ratio columns computed for {symbol}")
                return None

            logger.info(
                f"Fetched {len(fin_t)} quarterly reports for {symbol}; "
                f"{len(ratio_df.columns)} ratio columns"
            )
            return ratio_df

        except Exception as e:
            logger.warning(f"Financial feature fetch failed for {symbol}: {e}")
            return None

    def _compute_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute normalised financial ratios, all prefixed with ``fin_``."""

        def col(*names: str) -> pd.Series | None:
            for n in names:
                if n in df.columns:
                    s = pd.to_numeric(df[n], errors="coerce")
                    if s.notna().any():
                        return s
            return None

        rev     = col("Total Revenue")
        ni      = col("Net Income",
                       "Net Income From Continuing Operations",
                       "Net Income Applicable To Common Shares")
        ebit    = col("EBIT", "Operating Income")
        ebitda  = col("EBITDA", "Normalized EBITDA")
        interest = col("Interest Expense", "Interest Expense Non Operating")
        debt    = col("Total Debt",
                       "Long Term Debt And Capital Lease Obligation",
                       "Long Term Debt")
        eq      = col("Stockholders Equity",
                       "Common Stockholders Equity",
                       "Total Equity Gross Minority Interest")
        ocf     = col("Operating Cash Flow",
                       "Cash Flow From Continuing Operating Activities")
        capex   = col("Capital Expenditure")
        ta      = col("Total Assets")

        out = pd.DataFrame(index=df.index)

        _eps = 1e-9  # avoid division by zero

        # Revenue growth QoQ and YoY (shifted over 1 and 4 quarters)
        if rev is not None:
            out["fin_revenue_growth_qoq"] = (
                rev.pct_change(1).replace([np.inf, -np.inf], np.nan)
            )
            out["fin_revenue_growth_yoy"] = (
                rev.pct_change(4).replace([np.inf, -np.inf], np.nan)
            )
            # Log-scaled revenue for absolute size signal
            out["fin_revenue_log"] = np.log1p(rev.clip(lower=0))

        # Profit margins
        if ni is not None and rev is not None:
            out["fin_net_margin"] = (
                ni / rev.replace(0, np.nan)
            ).clip(-5.0, 5.0)

        if ebitda is not None and rev is not None:
            out["fin_ebitda_margin"] = (
                ebitda / rev.replace(0, np.nan)
            ).clip(-5.0, 5.0)

        if ebit is not None and rev is not None:
            out["fin_operating_margin"] = (
                ebit / rev.replace(0, np.nan)
            ).clip(-5.0, 5.0)

        # Earnings growth QoQ and YoY
        if ni is not None:
            out["fin_earnings_growth_qoq"] = (
                ni.pct_change(1).replace([np.inf, -np.inf], np.nan)
            )
            out["fin_earnings_growth_yoy"] = (
                ni.pct_change(4).replace([np.inf, -np.inf], np.nan)
            )

        # Leverage
        if debt is not None and eq is not None:
            out["fin_debt_to_equity"] = (
                debt / eq.replace(0, np.nan)
            ).clip(-20.0, 20.0)

        if debt is not None and ta is not None:
            out["fin_debt_to_assets"] = (
                debt / ta.replace(0, np.nan)
            ).clip(0.0, 5.0)

        # Return on equity (trailing twelve months = rolling 4-quarter sum)
        if ni is not None and eq is not None:
            ni_ttm = ni.rolling(4, min_periods=1).sum()
            out["fin_roe_ttm"] = (ni_ttm / eq.replace(0, np.nan)).clip(-10.0, 10.0)

        # Cash-flow quality: operating cash flow vs net income
        if ocf is not None and ni is not None:
            out["fin_cashflow_quality"] = (
                ocf / ni.replace(0, np.nan)
            ).clip(-10.0, 10.0)

        if ocf is not None and rev is not None:
            out["fin_ocf_margin"] = (
                ocf / rev.replace(0, np.nan)
            ).clip(-5.0, 5.0)

        # Free cash flow margin (capex is usually negative in yfinance)
        if ocf is not None and capex is not None and rev is not None:
            fcf = ocf + capex
            out["fin_fcf_margin"] = (
                fcf / rev.replace(0, np.nan)
            ).clip(-5.0, 5.0)

        # Interest coverage: EBIT / |interest expense|
        if ebit is not None and interest is not None:
            out["fin_interest_coverage"] = (
                ebit / interest.replace(0, np.nan).abs()
            ).clip(-50.0, 50.0)

        # EPS surprise (merged from earnings_dates if available)
        if "eps_surprise_pct" in df.columns:
            out["fin_eps_surprise"] = (
                pd.to_numeric(df["eps_surprise_pct"], errors="coerce").clip(-2.0, 2.0)
            )

        out = out.replace([np.inf, -np.inf], np.nan)

        if out.empty or out.columns.empty:
            return pd.DataFrame()
        return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _strip_tz_index(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _safe_fetch(ticker, attr: str) -> pd.DataFrame | None:
    try:
        val = getattr(ticker, attr, None)
        if val is None or (hasattr(val, "empty") and val.empty):
            return None
        return val
    except Exception:
        return None


def _build_announcement_map(
    quarter_end_dates: pd.DatetimeIndex,
    earnings_dates_df: pd.DataFrame | None,
    lag_days: int,
) -> dict:
    """Map each fiscal quarter-end date → actual announcement date.

    Uses the earliest ``earnings_dates`` entry that falls in the window
    [quarter_end, quarter_end + 120 days].  Falls back to quarter_end +
    ``lag_days`` when no match is found.
    """
    # Build a sorted array of actual announcement dates
    actual: pd.DatetimeIndex | None = None
    if earnings_dates_df is not None and not earnings_dates_df.empty:
        try:
            actual = _strip_tz_index(pd.DatetimeIndex(earnings_dates_df.index))
            actual = actual.sort_values()
        except Exception:
            actual = None

    ann_map: dict = {}
    for qend in quarter_end_dates:
        qend_n = qend.tz_localize(None) if getattr(qend, "tzinfo", None) else qend
        matched = False
        if actual is not None and len(actual) > 0:
            win_end = qend_n + pd.Timedelta(days=120)
            cands = actual[(actual >= qend_n) & (actual <= win_end)]
            if len(cands) > 0:
                ann_map[qend] = cands[0]   # earliest date in window
                matched = True
        if not matched:
            ann_map[qend] = qend_n + pd.Timedelta(days=lag_days)

    return ann_map


def _extract_eps_surprise(ed: pd.DataFrame | None) -> pd.DataFrame | None:
    """Extract EPS surprise percentage from yfinance earnings_dates DataFrame."""
    if ed is None or ed.empty:
        return None
    try:
        ed_clean = ed.copy()
        ed_clean.index = _strip_tz_index(pd.DatetimeIndex(ed_clean.index))
        ed_clean = ed_clean.sort_index()

        col_map = {c.lower().replace(" ", "_").replace("(", "").replace(")", ""): c
                   for c in ed_clean.columns}

        # Try surprise% column first
        surprise_key = next((k for k in col_map if "surprise" in k and "%" in k), None)
        estimate_key = next((k for k in col_map if "estimate" in k), None)
        actual_key   = next((k for k in col_map if "reported" in k or
                             ("eps" in k and "estimate" not in k and "surprise" not in k)), None)

        result = pd.DataFrame(index=ed_clean.index)

        if surprise_key:
            pct = pd.to_numeric(ed_clean[col_map[surprise_key]], errors="coerce") / 100.0
            result["eps_surprise_pct"] = pct.clip(-2.0, 2.0)
        elif estimate_key and actual_key:
            est = pd.to_numeric(ed_clean[col_map[estimate_key]], errors="coerce")
            act = pd.to_numeric(ed_clean[col_map[actual_key]],   errors="coerce")
            pct = ((act - est) / est.abs().replace(0, np.nan)).clip(-2.0, 2.0)
            result["eps_surprise_pct"] = pct

        return result if not result.empty else None
    except Exception:
        return None
