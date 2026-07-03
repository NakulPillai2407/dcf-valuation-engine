"""
plotting.py
============
All Plotly chart construction for the DCF Valuation Engine lives here. Every
function takes already-computed data (DataFrames, lists, dicts of numbers)
and returns a `plotly.graph_objects.Figure` - no Streamlit calls, no
financial maths. `app.py` is responsible for calling `st.plotly_chart(fig)`.

Colour convention used throughout: green signals undervalued / positive /
upside; red signals overvalued / negative / downside. This mirrors how
equity research and trading desks conventionally colour-code valuation
output, and is applied consistently so a recruiter skimming the app can
read the charts at a glance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

COLOR_UNDERVALUED = "#2ECC71"
COLOR_OVERVALUED = "#E74C3C"
COLOR_NEUTRAL = "#3498DB"
COLOR_HISTORICAL = "#95A5A6"
COLOR_PROJECTED = "#3498DB"
COLOR_TERMINAL = "#9B59B6"


def plot_historical_financials(income_df: pd.DataFrame, fcff_df: pd.DataFrame) -> go.Figure:
    """Bar chart of 5 years of Revenue, EBIT and FCFF, with EBIT margin % overlaid.

    Gives the user a quick visual read on the company's historical
    trajectory before we start projecting anything: is revenue growing, is
    EBIT keeping pace (i.e. is the margin stable, expanding, or eroding),
    and how much of that operating profit actually converts into free cash
    flow once CapEx and working capital needs are accounted for.
    """
    years = income_df["date"].dt.year.astype(str)
    ebit_margin_pct = (income_df["ebit"] / income_df["revenue"]) * 100

    fig = go.Figure()
    fig.add_bar(name="Revenue", x=years, y=income_df["revenue"], marker_color="#2C3E50")
    fig.add_bar(name="EBIT", x=years, y=income_df["ebit"], marker_color="#2980B9")

    # FCFF may cover fewer years than the income statement (inner-join
    # dependent on cash flow statement availability), so align by year label.
    fcff_years = fcff_df["date"].dt.year.astype(str)
    fig.add_bar(name="FCFF", x=fcff_years, y=fcff_df["fcff"], marker_color="#27AE60")

    fig.add_trace(
        go.Scatter(
            name="EBIT Margin %",
            x=years,
            y=ebit_margin_pct,
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#E67E22", width=3),
        )
    )

    fig.update_layout(
        barmode="group",
        title="Historical Revenue, EBIT & FCFF",
        yaxis=dict(title="USD"),
        yaxis2=dict(title="EBIT Margin (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
    )
    return fig


def plot_revenue_projection(
    historical_dates: Sequence, historical_revenues: Sequence[float], projected_revenues: Sequence[float]
) -> go.Figure:
    """Line chart bridging historical (solid) and projected (dashed) revenue.

    The projection period is shaded so it's immediately visually obvious
    where actual filed results end and where model assumptions take over -
    an important distinction for anyone evaluating how much to trust the
    numbers.
    """
    historical_revenues = list(historical_revenues)
    historical_years = [pd.Timestamp(d).year for d in historical_dates]
    last_historical_year = historical_years[-1]
    projected_years = list(range(last_historical_year + 1, last_historical_year + 1 + len(projected_revenues)))

    # Bridge point: repeat the last historical value as the first point of
    # the projected series so the dashed line connects seamlessly.
    bridge_years = [last_historical_year] + projected_years
    bridge_revenues = [historical_revenues[-1]] + list(projected_revenues)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historical_years,
            y=historical_revenues,
            mode="lines+markers",
            name="Historical Revenue",
            line=dict(color=COLOR_HISTORICAL, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bridge_years,
            y=bridge_revenues,
            mode="lines+markers",
            name="Projected Revenue",
            line=dict(color=COLOR_PROJECTED, width=3, dash="dash"),
        )
    )

    fig.add_vrect(
        x0=last_historical_year,
        x1=projected_years[-1],
        fillcolor=COLOR_PROJECTED,
        opacity=0.08,
        line_width=0,
        annotation_text="Projection Period",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Revenue: Historical vs Projected",
        xaxis_title="Fiscal Year",
        yaxis_title="Revenue (USD)",
        template="plotly_white",
    )
    return fig


def plot_fcff_waterfall(fcff_components: Dict[str, float], year_label: str) -> go.Figure:
    """Waterfall chart showing how FCFF is built up for a single projected year.

    Sequence shown: NOPAT (EBIT after tax) -> + D&A -> - CapEx -> - Change
    in Working Capital -> = FCFF total. This is the most intuitive way to
    explain WHY free cash flow differs from accounting profit: it makes the
    non-cash add-backs and cash reinvestment deductions visually explicit
    rather than burying them in a formula.

    Parameters
    ----------
    fcff_components : dict with keys "NOPAT", "D&A", "CapEx", "Change in WC"
                       (D&A positive/add, CapEx and Change in WC as the
                       magnitudes to be subtracted)
    year_label       : e.g. "Year 1 (FY2027)" for the chart title
    """
    fig = go.Figure(
        go.Waterfall(
            name="FCFF Build-up",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["NOPAT", "+ D&A", "- CapEx", "- Change in WC", "= FCFF"],
            y=[
                fcff_components["NOPAT"],
                fcff_components["D&A"],
                -fcff_components["CapEx"],
                -fcff_components["Change in WC"],
                0,  # total bar computes itself from the running sum
            ],
            connector=dict(line=dict(color="rgb(120,120,120)")),
            increasing=dict(marker=dict(color="#27AE60")),
            decreasing=dict(marker=dict(color="#E74C3C")),
            totals=dict(marker=dict(color="#2C3E50")),
        )
    )
    fig.update_layout(
        title=f"FCFF Build-up — {year_label}",
        yaxis_title="USD",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def plot_dcf_bridge(
    pv_fcffs: Sequence[float], pv_terminal_value: float, net_debt: float
) -> go.Figure:
    """The headline DCF bridge waterfall: PV(FCFF years) + PV(Terminal Value)
    = Enterprise Value -> - Net Debt = Equity Value.

    This is the single most important chart in the app - it shows visually
    exactly how the final equity value was built, step by step, from the
    discounted projection-period cash flows through to the terminal value,
    and then the financing bridge (subtracting net debt) down to what
    belongs to shareholders.
    """
    labels = [f"PV Year {i+1}" for i in range(len(pv_fcffs))]
    labels += ["PV Terminal Value", "Enterprise Value", "Less: Net Debt", "Equity Value"]

    measures = ["relative"] * len(pv_fcffs) + ["relative", "total", "relative", "total"]
    values = list(pv_fcffs) + [pv_terminal_value, 0, -net_debt, 0]

    fig = go.Figure(
        go.Waterfall(
            name="DCF Bridge",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector=dict(line=dict(color="rgb(120,120,120)")),
            increasing=dict(marker=dict(color="#27AE60")),
            decreasing=dict(marker=dict(color="#E74C3C")),
            totals=dict(marker=dict(color="#2C3E50")),
        )
    )
    fig.update_layout(
        title="DCF Bridge: Present Values → Enterprise Value → Equity Value",
        yaxis_title="USD",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def plot_pv_breakdown_pie(pv_fcffs: Sequence[float], pv_terminal_value: float) -> go.Figure:
    """Pie chart: % of Enterprise Value from the projection period vs terminal value.

    If the terminal value slice exceeds ~80% of the total, the valuation
    rests mostly on an assumption about the distant, unmodelled future
    (perpetual growth at rate g) rather than on near-term, explicitly
    forecast cash flows - worth flagging directly to the user as a
    reliability caveat.
    """
    projection_total = sum(pv_fcffs)
    fig = go.Figure(
        go.Pie(
            labels=["PV of Projection Period FCFF", "PV of Terminal Value"],
            values=[projection_total, pv_terminal_value],
            marker=dict(colors=[COLOR_PROJECTED, COLOR_TERMINAL]),
            hole=0.45,
        )
    )
    fig.update_layout(title="Enterprise Value Composition", template="plotly_white")
    return fig


def plot_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame, title: str, current_price: float
) -> go.Figure:
    """Colour-coded heatmap of a sensitivity grid (e.g. WACC vs terminal growth).

    Green cells indicate assumption combinations where the DCF intrinsic
    value exceeds the current market price (implying the stock looks
    undervalued under those assumptions); red cells indicate the opposite.
    A diverging colour scale centred on the current price makes it
    immediately visually clear how much of the assumption space supports a
    "buy" versus "sell" conclusion, rather than relying on a single point
    estimate.
    """
    z = sensitivity_df.values.astype(float)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(sensitivity_df.columns),
            y=list(sensitivity_df.index),
            colorscale=[[0, COLOR_OVERVALUED], [0.5, "#F5F5F5"], [1, COLOR_UNDERVALUED]],
            zmid=current_price,
            colorbar=dict(title="Value/Share"),
            text=[[f"${v:,.0f}" if pd.notna(v) else "" for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: $%{z:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{title} (current price: ${current_price:,.2f})",
        xaxis_title=sensitivity_df.columns.name or "",
        yaxis_title=sensitivity_df.index.name or "",
        template="plotly_white",
    )
    return fig


def plot_scenario_comparison(scenario_dict: Dict[str, float], current_price: float) -> go.Figure:
    """Horizontal bar chart comparing Bull/Base/Bear intrinsic values to current price.

    A single vertical reference line for the current market price lets the
    viewer instantly see, across all three named scenarios, whether the
    stock would need a bearish, base-case, or bullish set of assumptions to
    be considered fairly valued at today's price.
    """
    scenarios = list(scenario_dict.keys())
    values = list(scenario_dict.values())
    colors = [
        COLOR_UNDERVALUED if v >= current_price else COLOR_OVERVALUED for v in values
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=scenarios,
            orientation="h",
            marker_color=colors,
            text=[f"${v:,.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Current Price: ${current_price:,.2f}",
        annotation_position="top",
    )
    fig.update_layout(
        title="Scenario Comparison: Intrinsic Value vs Current Price",
        xaxis_title="Intrinsic Value per Share (USD)",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def plot_valuation_gauge(intrinsic_value: float, current_price: float) -> go.Figure:
    """Gauge chart showing current price positioned within an undervalued/overvalued range.

    The gauge range is centred on the intrinsic value (+/- 50%), with the
    needle placed at the current market price and colour zones running deep
    green (heavily undervalued, price far below intrinsic value) through to
    red (heavily overvalued, price far above intrinsic value). The
    upside/downside percentage is displayed as the gauge's number readout.
    """
    upside_pct = (intrinsic_value - current_price) / current_price * 100

    gauge_min = intrinsic_value * 0.5
    gauge_max = intrinsic_value * 1.5

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_price,
            number={"prefix": "$", "valueformat": ",.2f"},
            delta={
                "reference": intrinsic_value,
                "valueformat": ",.2f",
                "decreasing": {"color": COLOR_UNDERVALUED},
                "increasing": {"color": COLOR_OVERVALUED},
            },
            title={"text": f"Current Price vs Intrinsic Value (${intrinsic_value:,.2f})<br>"
                            f"Upside/Downside: {upside_pct:+.1f}%"},
            gauge={
                "axis": {"range": [gauge_min, gauge_max]},
                "bar": {"color": "#2C3E50"},
                "steps": [
                    {"range": [gauge_min, intrinsic_value * 0.8], "color": "#1E8449"},
                    {"range": [intrinsic_value * 0.8, intrinsic_value * 0.95], "color": "#82E0AA"},
                    {"range": [intrinsic_value * 0.95, intrinsic_value * 1.05], "color": "#F7F9F9"},
                    {"range": [intrinsic_value * 1.05, intrinsic_value * 1.2], "color": "#F1948A"},
                    {"range": [intrinsic_value * 1.2, gauge_max], "color": "#C0392B"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.9,
                    "value": intrinsic_value,
                },
            },
        )
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def plot_historical_price_vs_dcf(
    price_history: pd.DataFrame,
    intrinsic_value: float,
    analyst_target: Optional[float] = None,
) -> go.Figure:
    """2-year historical price chart with DCF intrinsic value and analyst target overlaid.

    Important framing note shown in the chart title: the DCF intrinsic
    value is a LONG-RUN estimate of fundamental worth, not a short-term
    price target - it should not be expected to track the day-to-day price
    line, and divergence between the two is normal and expected, not
    necessarily a sign of model error.

    Parameters
    ----------
    price_history   : DataFrame with columns "Date", "Close" (see
                       data_fetcher.fetch_price_history)
    intrinsic_value : our DCF's per-share estimate, shown as a horizontal
                       dashed line
    analyst_target  : optional consensus analyst target, shown as a second
                       horizontal dashed line if provided
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_history["Date"],
            y=price_history["Close"],
            mode="lines",
            name="Historical Price",
            line=dict(color="#2C3E50", width=2),
        )
    )
    fig.add_hline(
        y=intrinsic_value,
        line_dash="dash",
        line_color=COLOR_UNDERVALUED,
        annotation_text=f"DCF Intrinsic Value: ${intrinsic_value:,.2f}",
        annotation_position="top left",
    )
    if analyst_target:
        fig.add_hline(
            y=analyst_target,
            line_dash="dot",
            line_color="#F39C12",
            annotation_text=f"Analyst Consensus Target: ${analyst_target:,.2f}",
            annotation_position="bottom left",
        )
    fig.update_layout(
        title="2-Year Price History vs DCF Intrinsic Value (a long-run estimate, not a short-term target)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
    )
    return fig
