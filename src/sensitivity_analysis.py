"""
sensitivity_analysis.py
========================
A single base-case DCF output is really just ONE point estimate drawn from
an entire range of plausible outcomes, since every input (growth, margins,
WACC, terminal growth) is an assumption, not a certainty. This module runs
the full DCF repeatedly across a GRID of assumptions to show how sensitive
the final intrinsic value is to the two inputs that matter most - and also
runs three named "story" scenarios (bull / base / bear) as an intuitive
summary for a non-technical reader.

Every function here re-uses `dcf_model.run_full_dcf` as the single source
of truth for the DCF maths, so the sensitivity tables are always internally
consistent with the headline result shown elsewhere in the app.
"""

from __future__ import annotations

import copy
from typing import Dict, Sequence

import pandas as pd

from src.dcf_model import run_full_dcf


def _dcf_value_for(
    base_assumptions: Dict,
    historical_data: Dict,
    shares: float,
    net_debt: float,
    wacc: float,
    overrides: Dict,
) -> float:
    """Run one full DCF with `overrides` applied on top of the base assumptions.

    Returns just the intrinsic value per share (a float), or None if the
    combination of inputs is invalid (e.g. terminal growth >= WACC for that
    grid cell) - invalid cells are left blank in the sensitivity table
    rather than crashing the whole grid.
    """
    assumptions = copy.deepcopy(base_assumptions)
    assumptions.update(overrides)
    try:
        result = run_full_dcf(assumptions, historical_data, wacc, shares, net_debt)
        return result["intrinsic_value_per_share"]
    except ValueError:
        # Terminal growth >= WACC for this particular grid cell - undefined,
        # shown as a blank cell in the heatmap rather than raising.
        return None


def wacc_vs_growth_sensitivity(
    base_assumptions: Dict,
    wacc_range: Sequence[float],
    growth_range: Sequence[float],
    historical_data: Dict,
    shares: float,
    net_debt: float,
) -> pd.DataFrame:
    """Build a grid of intrinsic value per share across WACC x terminal growth rate.

    This is the single most important sensitivity table in any DCF: WACC
    and terminal growth rate (g) are the two assumptions the final
    valuation is most sensitive to (because the terminal value formula is
    FCFF * (1+g) / (WACC - g), a ratio that can swing dramatically for
    small changes in either input). The headline "base case" number shown
    elsewhere in the app is really just ONE cell in this grid - everything
    else here shows how fragile (or robust) that number is to reasonably
    plausible variation in these two assumptions.

    Parameters
    ----------
    base_assumptions : the assumptions dict (see dcf_model.run_full_dcf) to
                        vary WACC/growth on top of - all other assumptions
                        (revenue growth, margins, etc.) are held fixed
    wacc_range       : sequence of WACC values to test, e.g. np.arange(0.06, 0.125, 0.005)
    growth_range     : sequence of terminal growth rates to test, e.g. np.arange(0.01, 0.045, 0.005)
    historical_data, shares, net_debt : passed straight through to run_full_dcf

    Returns
    -------
    pd.DataFrame where the ROW index is WACC (formatted as % strings) and
    the COLUMN index is terminal growth rate (formatted as % strings), and
    each cell is the resulting intrinsic value per share. Cells where
    growth >= WACC are NaN (undefined).
    """
    rows = {}
    for wacc in wacc_range:
        row = {}
        for g in growth_range:
            value = _dcf_value_for(
                base_assumptions,
                historical_data,
                shares,
                net_debt,
                wacc=wacc,
                overrides={"terminal_growth_rate": g},
            )
            row[f"{g:.1%}"] = value
        rows[f"{wacc:.1%}"] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "WACC"
    df.columns.name = "Terminal Growth Rate"
    return df


def margin_vs_growth_sensitivity(
    base_assumptions: Dict,
    margin_range: Sequence[float],
    growth_range: Sequence[float],
    wacc: float,
    historical_data: Dict,
    shares: float,
    net_debt: float,
) -> pd.DataFrame:
    """Build a grid of intrinsic value per share across EBIT margin x revenue growth rate.

    While the WACC/terminal-growth grid shows sensitivity to FINANCIAL
    assumptions (the discount rate and long-run economic growth), this grid
    shows sensitivity to OPERATIONAL assumptions - how profitable the
    business is (EBIT margin) and how fast it can grow revenue in the
    explicit projection window. This is useful for understanding how much
    of the valuation depends on the company successfully executing
    operationally (expanding margins, sustaining growth) versus purely
    financial/macro factors.

    Note: this varies the near-term (year 1-N) revenue growth rate
    uniformly across all projection years for simplicity - the base case
    step-down profile (different early vs late growth) is held at the ratio
    implied by `base_assumptions["growth_rates"]` scaled by the tested
    growth rate's ratio to the base first-year rate, keeping the grid
    tractable while still respecting the "growth decelerates over time"
    principle.

    Parameters
    ----------
    base_assumptions : base DCF assumptions dict; margin_range/growth_range
                        override "ebit_margin" and "growth_rates"
    margin_range     : sequence of EBIT margin values to test, e.g. 0.10 to 0.35
    growth_range     : sequence of revenue growth rates to test, e.g. 0.02 to 0.25
    wacc             : held constant across this grid (unlike the WACC/growth grid)
    historical_data, shares, net_debt : passed straight through to run_full_dcf

    Returns
    -------
    pd.DataFrame where rows = EBIT margin (% strings), columns = revenue
    growth rate (% strings), cells = intrinsic value per share.
    """
    n_years = len(base_assumptions["growth_rates"])

    rows = {}
    for margin in margin_range:
        row = {}
        for growth in growth_range:
            # Apply a flat growth rate across every projection year for
            # this grid cell - simpler than replicating a full step-down
            # schedule, and sufficient to show directional sensitivity.
            flat_growth_rates = [growth] * n_years
            value = _dcf_value_for(
                base_assumptions,
                historical_data,
                shares,
                net_debt,
                wacc=wacc,
                overrides={"ebit_margin": margin, "growth_rates": flat_growth_rates},
            )
            row[f"{growth:.1%}"] = value
        rows[f"{margin:.1%}"] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "EBIT Margin"
    df.columns.name = "Revenue Growth Rate"
    return df


def scenario_analysis(
    base_assumptions: Dict,
    wacc: float,
    historical_data: Dict,
    shares: float,
    net_debt: float,
) -> Dict[str, float]:
    """Run Bull / Base / Bear named scenarios and return their intrinsic values.

    Sensitivity grids are precise but can be overwhelming for a
    non-technical reader; three named scenarios translate the same
    underlying sensitivity into an intuitive story:

      - Bull case : top-quartile revenue growth, expanding EBIT margins, and
                    a LOWER discount rate (reflecting lower perceived risk
                    in an optimistic environment) - everything that could
                    reasonably go right, going right.
      - Base case : the model's current assumptions exactly as configured
                    by the user (typically anchored to historical averages).
      - Bear case : slowing growth, compressing margins, and a HIGHER
                    discount rate (reflecting higher perceived risk) -
                    everything that could reasonably go wrong, going wrong.

    The bull/bear adjustments below are deliberately modest, mechanical
    perturbations (not hand-tuned per company) so the scenario spread is
    reproducible and explainable: growth +/-30% relative, EBIT margin
    +/-2 percentage points, WACC -/+1 percentage point, terminal growth
    +/-0.5 percentage points.

    Returns
    -------
    dict: {"Bull Case": float, "Base Case": float, "Bear Case": float},
    each value being the intrinsic value per share under that scenario.
    """
    base_growth_rates = base_assumptions["growth_rates"]

    bull_assumptions = copy.deepcopy(base_assumptions)
    bull_assumptions["growth_rates"] = [g * 1.3 for g in base_growth_rates]
    bull_assumptions["ebit_margin"] = base_assumptions["ebit_margin"] + 0.02
    bull_assumptions["terminal_growth_rate"] = min(
        base_assumptions["terminal_growth_rate"] + 0.005, wacc - 0.005
    )
    bull_wacc = max(wacc - 0.01, 0.01)

    bear_assumptions = copy.deepcopy(base_assumptions)
    bear_assumptions["growth_rates"] = [g * 0.7 for g in base_growth_rates]
    bear_assumptions["ebit_margin"] = max(base_assumptions["ebit_margin"] - 0.02, 0.0)
    bear_assumptions["terminal_growth_rate"] = max(
        base_assumptions["terminal_growth_rate"] - 0.005, 0.0
    )
    bear_wacc = wacc + 0.01

    bull_value = run_full_dcf(
        bull_assumptions, historical_data, bull_wacc, shares, net_debt
    )["intrinsic_value_per_share"]
    base_value = run_full_dcf(
        base_assumptions, historical_data, wacc, shares, net_debt
    )["intrinsic_value_per_share"]
    bear_value = run_full_dcf(
        bear_assumptions, historical_data, bear_wacc, shares, net_debt
    )["intrinsic_value_per_share"]

    return {
        "Bull Case": bull_value,
        "Base Case": base_value,
        "Bear Case": bear_value,
    }
