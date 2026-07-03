"""
financial_processor.py
=======================
Turns the raw statement DataFrames produced by `data_fetcher.py` into
analysis-ready figures: historical Free Cash Flow to Firm (FCFF), the
historical average margins/ratios used as DEFAULT projection assumptions,
and net debt.

Nothing here talks to an external API or to Streamlit - this module is pure
data transformation and financial arithmetic on already-fetched DataFrames.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_fcff(
    income_df: pd.DataFrame, cashflow_df: pd.DataFrame, balance_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute historical Free Cash Flow to Firm (FCFF) for each fiscal year.

        FCFF = EBIT * (1 - Tax Rate) + D&A - CapEx - Change in Working Capital

    Financial intuition for each term:
      - EBIT * (1 - Tax Rate) = NOPAT (Net Operating Profit After Tax). We
        start from EBIT (not net income) because EBIT is calculated BEFORE
        interest expense, i.e. before any financing decisions. This makes
        FCFF capital-structure neutral - it represents the cash the
        underlying BUSINESS generates for ALL capital providers (both debt
        and equity holders), not just what's left over for shareholders
        after debt payments. We still deduct tax because the government's
        claim on profit is not optional and does not depend on financing.
      - + D&A: depreciation and amortisation reduced EBIT as an accounting
        expense, but no cash actually left the business when that expense
        was recorded (the cash left years earlier when the asset was
        purchased). We add it back to undo this non-cash accounting effect.
      - - CapEx: conversely, capital expenditure IS a real cash outflow in
        the year it happens, even though accounting spreads ("depreciates")
        the expense over many future years. We subtract the full cash cost
        here because we're building a CASH flow, not an accounting profit.
      - - Change in Working Capital: an increase in working capital (e.g.
        more inventory or more money tied up in unpaid customer invoices)
        consumes cash even though it may not show up as an expense on the
        income statement. We subtract increases (and add back decreases).

    Parameters
    ----------
    income_df   : output of fetch_income_statements (needs date, ebit,
                  effective_tax_rate)
    cashflow_df : output of fetch_cash_flow_statements (needs date, capex,
                  D&A, change in working capital)
    balance_df  : accepted for interface symmetry with the rest of the
                  pipeline; not currently required for FCFF itself but kept
                  so callers can extend this function with balance-sheet
                  driven adjustments later without changing the signature.

    Returns
    -------
    pd.DataFrame indexed by date with columns: ebit, tax_rate, da, capex,
    change_in_wc, nopat, fcff. Years are matched on `date`; any year present
    in one statement but missing from another is dropped (inner join) since
    a partial FCFF figure would be misleading.
    """
    merged = pd.merge(
        income_df[["date", "ebit", "effective_tax_rate"]],
        cashflow_df[
            [
                "date",
                "capitalExpenditure",
                "depreciationAndAmortization",
                "changeInWorkingCapital",
            ]
        ],
        on="date",
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No overlapping fiscal years between income statement and cash flow "
            "statement data - cannot compute historical FCFF."
        )

    # Fall back to a sensible statutory-ish tax rate if a given year's
    # effective tax rate is missing or nonsensical (e.g. negative pretax
    # income can produce a distorted or negative "effective" rate).
    fallback_tax_rate = merged["effective_tax_rate"].dropna()
    fallback_tax_rate = (
        fallback_tax_rate[(fallback_tax_rate >= 0) & (fallback_tax_rate <= 0.6)].mean()
        if not fallback_tax_rate.empty
        else 0.21
    )
    tax_rate = merged["effective_tax_rate"].apply(
        lambda x: x if (x is not None and 0 <= x <= 0.6) else fallback_tax_rate
    )

    nopat = merged["ebit"] * (1 - tax_rate)
    da = merged["depreciationAndAmortization"].fillna(0)
    capex = merged["capitalExpenditure"].fillna(0)
    change_in_wc = merged["changeInWorkingCapital"].fillna(0)

    fcff = nopat + da - capex - change_in_wc

    result = pd.DataFrame(
        {
            "date": merged["date"],
            "ebit": merged["ebit"],
            "tax_rate": tax_rate,
            "da": da,
            "capex": capex,
            "change_in_wc": change_in_wc,
            "nopat": nopat,
            "fcff": fcff,
        }
    ).sort_values("date").reset_index(drop=True)

    return result


def compute_historical_margins(
    income_df: pd.DataFrame, cashflow_df: pd.DataFrame
) -> Dict[str, float]:
    """Compute 5-year historical averages used as DEFAULT projection sliders.

    We use historical averages as a starting point because they reflect the
    company's ACTUAL, demonstrated operating efficiency - a real, observed
    track record rather than a guess. However, history is not destiny:
    users should adjust these defaults based on forward-looking factors the
    historical average cannot capture, such as management guidance, new
    product launches, changing competitive dynamics, or macro/sector shifts
    (e.g. a maturing company should probably NOT be projected to keep
    growing at its high-growth-era historical average).

    Computed metrics (all as decimals, e.g. 0.08 = 8%):
      - revenue_growth_rate  : average year-on-year revenue growth
      - ebit_margin          : average EBIT / Revenue
      - capex_as_pct_revenue : average CapEx / Revenue
      - da_as_pct_revenue    : average D&A / Revenue
      - effective_tax_rate   : average effective tax rate

    Returns a dict of the five values above. Raises no exceptions for
    missing individual years - uses whatever valid data is available and
    falls back to conservative defaults if a metric cannot be computed at
    all (e.g. only one year of revenue history, so growth can't be computed).
    """
    revenue = income_df["revenue"].astype(float)
    revenue_growth = revenue.pct_change().dropna()
    revenue_growth_rate = (
        float(revenue_growth.mean()) if not revenue_growth.empty else 0.05
    )

    ebit_margin_series = (income_df["ebit"] / income_df["revenue"]).dropna()
    ebit_margin = float(ebit_margin_series.mean()) if not ebit_margin_series.empty else 0.15

    # CapEx / D&A live on the cash flow statement, revenue on the income
    # statement - merge on date so each ratio uses the correct matching year.
    merged = pd.merge(
        income_df[["date", "revenue"]], cashflow_df, on="date", how="inner"
    )
    capex_pct_series = (merged["capitalExpenditure"] / merged["revenue"]).dropna()
    capex_as_pct_revenue = (
        float(capex_pct_series.mean()) if not capex_pct_series.empty else 0.05
    )

    da_pct_series = (merged["depreciationAndAmortization"] / merged["revenue"]).dropna()
    da_as_pct_revenue = float(da_pct_series.mean()) if not da_pct_series.empty else 0.03

    tax_rate_series = income_df["effective_tax_rate"].dropna()
    tax_rate_series = tax_rate_series[(tax_rate_series >= 0) & (tax_rate_series <= 0.6)]
    effective_tax_rate = (
        float(tax_rate_series.mean()) if not tax_rate_series.empty else 0.21
    )

    return {
        "revenue_growth_rate": revenue_growth_rate,
        "ebit_margin": ebit_margin,
        "capex_as_pct_revenue": capex_as_pct_revenue,
        "da_as_pct_revenue": da_as_pct_revenue,
        "effective_tax_rate": effective_tax_rate,
    }


def compute_net_debt(balance_df: pd.DataFrame) -> float:
    """Compute net debt from the most recent balance sheet.

        Net Debt = Total Debt - Cash & Cash Equivalents

    We use NET (not gross) debt when bridging Enterprise Value to Equity
    Value because cash sitting on the balance sheet is a real asset that
    could, in principle, be used immediately to pay down debt. Two
    companies with identical gross debt but different cash piles have very
    different real financial risk and different amounts left over for
    equity holders - net debt captures that difference; gross debt alone
    would not.

    Uses the LATEST available year in `balance_df` (assumed sorted ascending
    by date, consistent with the other fetch/process functions in this
    codebase).
    """
    if balance_df.empty:
        raise ValueError("Balance sheet data is empty - cannot compute net debt.")

    latest = balance_df.iloc[-1]
    total_debt = latest["totalDebt"] or 0
    cash = latest["cashAndCashEquivalents"] or 0
    return float(total_debt - cash)


def get_latest_shares_outstanding(balance_df: pd.DataFrame) -> float:
    """Return the most recent shares outstanding figure from the balance sheet.

    Used to convert total Equity Value into a per-share intrinsic value.
    """
    if balance_df.empty:
        raise ValueError("Balance sheet data is empty - cannot read shares outstanding.")
    shares = balance_df.iloc[-1]["sharesOutstanding"]
    if not shares or shares <= 0:
        raise ValueError(
            "Shares outstanding is missing or zero in the balance sheet data."
        )
    return float(shares)
