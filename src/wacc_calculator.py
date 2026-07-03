"""
wacc_calculator.py
===================
Computes the Weighted Average Cost of Capital (WACC) - the discount rate
used to bring every projected future cash flow back to a present value in
the DCF model.

WACC is arguably the single most influential number in a DCF: it is the
blended annual return that ALL of the company's capital providers (both
shareholders and lenders) require for the risk they are taking on. A higher
WACC discounts future cash flows more heavily and produces a LOWER
valuation; a lower WACC produces a HIGHER valuation. Because the terminal
value formula divides by (WACC - g), even a 1 percentage point change in
WACC can move the final intrinsic value by 20-30% - so every input here is
commented in detail.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def compute_cost_of_equity(
    beta: float, risk_free_rate: float, equity_risk_premium: float = 0.055
) -> float:
    """Compute the Cost of Equity (Re) using the Capital Asset Pricing Model.

        Re = Rf + beta * ERP

    Term by term:
      - Rf (risk_free_rate): the return an investor can earn with
        essentially zero risk, proxied by the 10-year US Treasury yield.
        This is the "floor" return - nobody would accept equity risk for
        less than what they could get risk-free.
      - beta: measures how much MORE (or less) volatile this specific stock
        is versus the overall market. beta = 1.0 means the stock tends to
        move in lockstep with the market. beta = 1.5 means that, on average,
        when the market moves 1%, this stock moves 1.5% (either direction) -
        i.e. it carries 50% more systematic (market-wide) risk than average.
        beta < 1.0 means the stock is comparatively defensive.
      - ERP (equity_risk_premium): the extra annual return investors demand,
        on average, for holding stocks (market-wide) instead of risk-free
        bonds. Prof. Aswath Damodaran (NYU Stern) publishes a widely-used
        estimate for the US market, around 5.5% as of recent years - used
        here as the default, but made user-adjustable since ERP is
        genuinely debated and varies by source/methodology.

    Result: the annualised return shareholders require to compensate them
    for the risk of owning this specific stock, given its market-wide risk
    exposure (beta) relative to the risk-free alternative.

    Parameters
    ----------
    beta                : the stock's beta (dimensionless, ~0.3-2.5 typical range)
    risk_free_rate      : decimal, e.g. 0.045 for 4.5%
    equity_risk_premium : decimal, e.g. 0.055 for 5.5% (Damodaran US default)

    Returns the cost of equity as a decimal (e.g. 0.09 for 9%).
    """
    return risk_free_rate + beta * equity_risk_premium


def compute_cost_of_debt(income_df: pd.DataFrame, balance_df: pd.DataFrame) -> float:
    """Compute the (pre-tax) Cost of Debt (Rd).

        Rd = Interest Expense / Average Total Debt

    We use the most recent year's interest expense divided by the average of
    the most recent two years' total debt (when available), since interest
    expense accrues over the period during which the debt balance may have
    changed - averaging the opening and closing balance gives a more
    representative "debt outstanding during the year" figure than either
    endpoint alone.

    This produces the BEFORE-TAX cost of debt. In the WACC formula itself we
    multiply this by (1 - tax rate) - NOT here - because interest payments
    are tax-deductible (the "interest tax shield"): every dollar of interest
    paid reduces taxable income by a dollar, so the government effectively
    subsidises part of the cost of debt. Keeping the tax adjustment out of
    this function keeps its output ("what lenders actually charge") separate
    from the capital-structure-level tax effect applied in compute_wacc().

    Falls back to the most recent single year of debt if only one year of
    balance sheet data is available.
    """
    if income_df.empty or balance_df.empty:
        raise ValueError("Income statement or balance sheet data is empty.")

    latest_interest_expense = income_df.iloc[-1]["interestExpense"] or 0

    if len(balance_df) >= 2:
        average_total_debt = (
            balance_df.iloc[-1]["totalDebt"] + balance_df.iloc[-2]["totalDebt"]
        ) / 2
    else:
        average_total_debt = balance_df.iloc[-1]["totalDebt"]

    if not average_total_debt or average_total_debt <= 0:
        # No meaningful debt load -> cost of debt is not economically
        # meaningful; return 0 rather than dividing by zero. Its weight
        # (debt_weight) in WACC will also be near zero in this case.
        return 0.0

    return float(latest_interest_expense / average_total_debt)


def compute_capital_structure(profile: Dict, balance_df: pd.DataFrame) -> Dict[str, float]:
    """Determine the weights of equity and debt in the capital structure.

        E / V  and  D / V,   where V = E + D

    We use the MARKET value of equity (market capitalisation = share price *
    shares outstanding) rather than the book value of equity shown on the
    balance sheet. Book equity reflects historical accounting entries
    (retained earnings, paid-in capital, etc.) and can be wildly different
    from what investors currently think the company is worth. WACC is a
    forward-looking discount rate, so it should reflect today's market-based
    view of value, not sunk historical cost.

    For debt we use the BOOK value of total debt as a practical proxy for
    market value of debt. Unlike equity, most corporate debt does not trade
    on a liquid public market, so an observable market price usually isn't
    available - book value (what's actually owed) is the standard
    workaround used in practitioner DCFs.

    Returns a dict with: equity_weight, debt_weight, market_cap, total_debt.
    """
    market_cap = profile.get("marketCap")
    if not market_cap or market_cap <= 0:
        raise ValueError(
            "Market capitalisation is missing or invalid - cannot compute capital "
            "structure weights for WACC."
        )

    total_debt = balance_df.iloc[-1]["totalDebt"] or 0
    total_capital = market_cap + total_debt

    if total_capital <= 0:
        raise ValueError("Total capital (equity + debt) is zero or negative.")

    return {
        "equity_weight": float(market_cap / total_capital),
        "debt_weight": float(total_debt / total_capital),
        "market_cap": float(market_cap),
        "total_debt": float(total_debt),
    }


def compute_wacc(
    cost_of_equity: float,
    cost_of_debt: float,
    equity_weight: float,
    debt_weight: float,
    tax_rate: float,
) -> float:
    """Compute the Weighted Average Cost of Capital.

        WACC = (E/V) * Re + (D/V) * Rd * (1 - Tax Rate)

    WACC blends the required return of EVERY capital provider - shareholders
    (Re) and lenders (Rd) - weighted by how much of the company's total
    capital each group provides (E/V and D/V respectively). The (1 - Tax
    Rate) term is applied only to the debt side because interest is
    tax-deductible: the effective, after-tax cost of a dollar of interest is
    lower than its stated rate because it shields other income from tax.
    Equity dividends/returns receive no equivalent deduction, so no tax
    adjustment is applied to the equity term.

    WACC is the single rate used to discount every future projected cash
    flow (and the terminal value) back to present value in Step 5 of the
    DCF. Because it compounds over many years and sits in the denominator
    of the terminal value formula (WACC - g), it is the most sensitive
    input in the entire model - a seemingly small 1 percentage point change
    can shift the final valuation by 20-30%. This is why the sensitivity
    analysis module treats WACC as one of its two primary axes.

    Returns WACC as a decimal (e.g. 0.085 for 8.5%).
    """
    return (equity_weight * cost_of_equity) + (
        debt_weight * cost_of_debt * (1 - tax_rate)
    )
