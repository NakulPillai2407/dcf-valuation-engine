"""
dcf_model.py
=============
The core DCF engine: projects future revenue and Free Cash Flow to Firm
(FCFF), computes the terminal value, discounts everything to present value,
and bridges Enterprise Value down to an intrinsic per-share equity value.

This module is pure financial mathematics - no I/O, no Streamlit, no
external API calls. Every function takes plain Python/NumPy inputs and
returns plain Python/NumPy outputs so it can be unit-tested in isolation and
reused by both the main app and the sensitivity analysis module (which
calls `run_full_dcf` repeatedly across a grid of assumptions).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union


def project_revenue(base_revenue: float, growth_rates: Union[float, Sequence[float]]) -> List[float]:
    """Project revenue forward one year at a time using a growth rate per year.

    A single high growth rate is rarely realistic to hold constant forever -
    real companies tend to see growth "step down" over time as they mature,
    face tougher year-on-year comparisons, and saturate their addressable
    market. This is why the app lets users specify a different growth rate
    for early projection years (e.g. years 1-3) versus later ones (e.g.
    years 4-5), rather than a single flat rate for the whole projection
    window.

    Parameters
    ----------
    base_revenue : the most recent actual (historical) revenue figure, i.e.
                   the starting point for year 1's projection
    growth_rates : either a single float applied to every projected year, or
                   a list/tuple with one growth rate per projected year
                   (its length determines the number of years projected)

    Returns
    -------
    List of projected revenue figures, one per year, in chronological order.
    Note: base_revenue itself is NOT included in the returned list - only
    the newly projected years are.
    """
    if isinstance(growth_rates, (int, float)):
        raise ValueError(
            "project_revenue requires an explicit number of years - pass a "
            "list of growth rates (e.g. [0.1] * 5), not a single float."
        )

    projected = []
    current = base_revenue
    for rate in growth_rates:
        current = current * (1 + rate)
        projected.append(current)
    return projected


def project_fcff(
    projected_revenues: Sequence[float],
    ebit_margin: float,
    tax_rate: float,
    capex_pct: float,
    da_pct: float,
    wc_change_pct: float = 0.01,
) -> List[float]:
    """Project Free Cash Flow to Firm for each future year from projected revenue.

    For each projected year t:

        FCFF_t = Revenue_t * ebit_margin * (1 - tax_rate)   [NOPAT]
                 + Revenue_t * da_pct                        [add back D&A]
                 - Revenue_t * capex_pct                      [less CapEx]
                 - Revenue_t * wc_change_pct                  [less change in WC]

    This mirrors the historical FCFF formula in financial_processor.py, but
    instead of using ACTUAL EBIT/D&A/CapEx figures from filed statements, it
    applies user-adjustable ASSUMPTIONS (expressed as percentages of
    revenue) to the PROJECTED revenue for each future year. Expressing every
    driver as a % of revenue is the standard simplification used in
    practitioner DCFs: it keeps the number of assumptions manageable (5
    ratios instead of 5 separate absolute-dollar forecasts) and scales
    sensibly as the projected business grows or shrinks.

    Parameters
    ----------
    projected_revenues : output of project_revenue() - one figure per year
    ebit_margin        : assumed EBIT / Revenue (decimal, e.g. 0.25)
    tax_rate            : assumed effective tax rate (decimal, e.g. 0.21)
    capex_pct           : assumed CapEx / Revenue (decimal, e.g. 0.05)
    da_pct              : assumed D&A / Revenue (decimal, e.g. 0.03)
    wc_change_pct       : assumed change in working capital / Revenue,
                          default 0.01 (1%) - a common simplifying
                          assumption when explicit working-capital forecasts
                          aren't available; growing companies typically need
                          to invest a small, steady percentage of revenue
                          growth into working capital (inventory, receivables)

    Returns
    -------
    List of projected FCFF figures, one per year, same length and order as
    `projected_revenues`.
    """
    fcff_list = []
    for revenue in projected_revenues:
        nopat = revenue * ebit_margin * (1 - tax_rate)
        da = revenue * da_pct
        capex = revenue * capex_pct
        change_in_wc = revenue * wc_change_pct
        fcff = nopat + da - capex - change_in_wc
        fcff_list.append(fcff)
    return fcff_list


def compute_terminal_value(final_fcff: float, terminal_growth_rate: float, wacc: float) -> float:
    """Compute the Terminal Value using the Gordon Growth (perpetuity) Model.

        TV = FCFF_final * (1 + g) / (WACC - g)

    A DCF cannot explicitly project cash flows forever, so after the
    explicit projection window (e.g. 5 years) we assume the business
    settles into a stable, mature state and grows its cash flow at a
    constant rate `g` FOREVER. The Gordon Growth Model is the standard
    closed-form formula for the present value (as of the end of the
    projection period) of a perpetuity that grows at a constant rate.

    For most companies the terminal value represents 60-80% of total
    Enterprise Value - which means the DCF's conclusion is often driven
    more by this single formula than by the explicit projection years. This
    is exactly why `g` must be chosen conservatively.

    Key constraint: WACC must be strictly greater than g. If g >= WACC, the
    perpetuity formula implies an infinite (or negative, nonsensical) value,
    because the cash flows would be assumed to grow faster than the rate at
    which we're discounting them. `g` should be close to long-run nominal
    GDP growth (roughly 2-3%) - no company can grow faster than the overall
    economy forever without eventually becoming the entire economy.

    Raises
    ------
    ValueError if terminal_growth_rate >= wacc, since the formula is
    mathematically undefined (or produces a nonsensical negative/infinite
    result) in that case.
    """
    if terminal_growth_rate >= wacc:
        raise ValueError(
            f"Terminal growth rate ({terminal_growth_rate:.2%}) must be strictly "
            f"less than WACC ({wacc:.2%}), otherwise the Gordon Growth Model "
            "produces an infinite or negative terminal value. Lower the terminal "
            "growth rate or increase WACC."
        )
    return final_fcff * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)


def discount_cash_flows(cash_flows: Sequence[float], wacc: float) -> List[float]:
    """Discount a series of future cash flows to their present value using WACC.

    For each cash flow at year t (t = 1, 2, 3, ... in order):

        PV_t = CashFlow_t / (1 + WACC)^t

    This is the fundamental "time value of money" calculation: a dollar
    received further in the future is worth less today, both because of the
    opportunity cost of not having that dollar to invest now, and because of
    the risk that the cash flow doesn't materialise as projected. WACC is
    used as the discount rate because it represents the return capital
    providers could otherwise earn elsewhere at similar risk.

    Parameters
    ----------
    cash_flows : list of cash flows, in chronological order, where the first
                 element is exactly ONE YEAR away (t=1), the second is TWO
                 years away (t=2), and so on
    wacc       : discount rate as a decimal (e.g. 0.09 for 9%)

    Returns
    -------
    List of present values, same length and order as `cash_flows`.
    """
    return [cf / ((1 + wacc) ** t) for t, cf in enumerate(cash_flows, start=1)]


def compute_enterprise_value(pv_fcffs: Sequence[float], pv_terminal_value: float) -> float:
    """Sum discounted projection-period cash flows and the discounted terminal value.

        Enterprise Value = sum(PV of each projected year's FCFF) + PV(Terminal Value)

    Enterprise Value represents the value of the ENTIRE operating business -
    the value available to be split between everyone who financed it (both
    debt holders and equity holders), independent of how it happens to be
    financed today. It is the sum of two pieces: the value generated during
    the years we explicitly modelled, plus the value of everything the
    business is expected to generate afterwards (the terminal value).

    For most real companies the terminal value component (see
    compute_terminal_value) dominates this sum - often 60-80% of the total -
    which is a useful sanity check to surface to the user (see
    tv_as_pct_of_ev in run_full_dcf).
    """
    return float(sum(pv_fcffs) + pv_terminal_value)


def compute_equity_value(enterprise_value: float, net_debt: float) -> float:
    """Bridge Enterprise Value down to Equity Value by subtracting net debt.

        Equity Value = Enterprise Value - Net Debt

    Enterprise Value belongs to ALL capital providers collectively - both
    lenders and shareholders. To find out what's left over specifically for
    SHAREHOLDERS, we must subtract what's owed to lenders first. We use NET
    debt (total debt minus cash) rather than gross debt because any cash
    sitting on the balance sheet could immediately be used to retire debt,
    so it effectively offsets the company's debt burden dollar-for-dollar.

    If net debt is negative (i.e. the company holds more cash than debt),
    this SUBTRACTS a negative number, meaning cash-rich companies see their
    Equity Value increase above Enterprise Value - which makes intuitive
    sense: excess cash is a direct benefit to shareholders.
    """
    return enterprise_value - net_debt


def compute_intrinsic_value_per_share(equity_value: float, shares_outstanding: float) -> float:
    """Convert total Equity Value into a per-share intrinsic value estimate.

        Intrinsic Value Per Share = Equity Value / Shares Outstanding

    This is the final output of the DCF, expressed in the same units as the
    market's quoted share price so it can be directly compared to determine
    whether the stock looks under- or over-valued.

    Raises ValueError if shares_outstanding is zero or negative, since that
    would produce a meaningless (or division-by-zero) result.
    """
    if shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be positive to compute per-share value.")
    return equity_value / shares_outstanding


def run_full_dcf(
    assumptions: Dict[str, float],
    historical_data: Dict[str, float],
    wacc: float,
    shares: float,
    net_debt: float,
) -> Dict:
    """Orchestrate the full 6-step DCF pipeline and return every intermediate result.

    This is the single entry point used by both the main app (for the
    user's chosen "base case" assumptions) and the sensitivity analysis
    module (which calls this repeatedly with varied assumptions across a
    grid). Keeping the whole pipeline in one function guarantees the
    sensitivity tables use EXACTLY the same maths as the headline result.

    Parameters
    ----------
    assumptions : dict with keys:
        - "base_revenue"            : most recent actual annual revenue
        - "projection_years"        : int, length of the explicit forecast window
        - "growth_rates"            : list of per-year revenue growth rates,
                                       one entry per projection year
        - "ebit_margin"             : assumed EBIT / Revenue
        - "tax_rate"                : assumed effective tax rate
        - "capex_pct"               : assumed CapEx / Revenue
        - "da_pct"                  : assumed D&A / Revenue
        - "terminal_growth_rate"    : assumed perpetual growth rate g
        - "wc_change_pct"           : optional, defaults to 0.01 if absent
    historical_data : accepted for interface symmetry / future extension
        (e.g. blending historical and projected figures in charts); not
        required by the maths in this function today.
    wacc     : discount rate to apply to every projected cash flow and the
               terminal value
    shares   : shares outstanding, used for the final per-share conversion
    net_debt : total debt minus cash, used to bridge EV to equity value

    Returns
    -------
    dict with keys: projected_revenues, projected_fcffs, pv_fcffs,
    terminal_value, pv_terminal_value, enterprise_value, equity_value,
    intrinsic_value_per_share, tv_as_pct_of_ev, projection_years.

    `tv_as_pct_of_ev` (terminal value as a % of enterprise value) is
    surfaced explicitly because it's the single best indicator of how much
    of the valuation rests on the terminal growth assumption versus
    explicitly modelled near-term cash flows - a number above ~80% signals
    a valuation that is highly sensitive to the choice of `g` and `wacc`.
    """
    growth_rates = assumptions["growth_rates"]
    projection_years = assumptions.get("projection_years", len(growth_rates))

    projected_revenues = project_revenue(assumptions["base_revenue"], growth_rates)

    projected_fcffs = project_fcff(
        projected_revenues,
        ebit_margin=assumptions["ebit_margin"],
        tax_rate=assumptions["tax_rate"],
        capex_pct=assumptions["capex_pct"],
        da_pct=assumptions["da_pct"],
        wc_change_pct=assumptions.get("wc_change_pct", 0.01),
    )

    pv_fcffs = discount_cash_flows(projected_fcffs, wacc)

    terminal_value = compute_terminal_value(
        projected_fcffs[-1], assumptions["terminal_growth_rate"], wacc
    )
    # The terminal value is computed AS OF the end of the projection period,
    # so it must be discounted back the same number of years as the final
    # projected cash flow (N years), not N+1.
    pv_terminal_value = terminal_value / ((1 + wacc) ** projection_years)

    enterprise_value = compute_enterprise_value(pv_fcffs, pv_terminal_value)
    equity_value = compute_equity_value(enterprise_value, net_debt)
    intrinsic_value_per_share = compute_intrinsic_value_per_share(equity_value, shares)

    tv_as_pct_of_ev = (
        pv_terminal_value / enterprise_value if enterprise_value != 0 else 0.0
    )

    return {
        "projected_revenues": projected_revenues,
        "projected_fcffs": projected_fcffs,
        "pv_fcffs": pv_fcffs,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "tv_as_pct_of_ev": tv_as_pct_of_ev,
        "projection_years": projection_years,
    }
