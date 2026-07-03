"""
app.py
=======
Main Streamlit application for the DCF Valuation Engine.

This file owns ALL Streamlit UI concerns (layout, session state, widgets,
caching of data-fetch calls) and orchestrates calls into the `src/` modules,
which contain all financial mathematics and data retrieval. No financial
formula is implemented directly in this file - every number shown here is
produced by a function in `src/`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src import data_fetcher as fetcher
from src import financial_processor as processor
from src import wacc_calculator as wacc_calc
from src import dcf_model as dcf
from src import sensitivity_analysis as sens
from src import plotting as plot

# Loads FMP_API_KEY from a local .env file (gitignored, never committed) so
# the key never has to be hardcoded into source or typed in every session.
load_dotenv()
ENV_API_KEY = os.getenv("FMP_API_KEY", "")

st.set_page_config(
    page_title="DCF Valuation Engine",
    page_icon="💰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Glossary & teaching helpers
#
# Every abbreviation used in the app's narrative text is defined here once.
# `tip()` turns a term into an inline hover-tooltip (an HTML <abbr> tag), and
# `glossary_box()` renders a small set of terms as styled reference cards
# inside a collapsible expander. `teach()` wraps a paragraph of explanatory
# text in a bordered callout so each tab reads as a guided walkthrough
# rather than a wall of numbers.
# ---------------------------------------------------------------------------

TERMS: dict[str, str] = {
    "Market Cap": "The total value of all a company's outstanding shares (share price × shares outstanding) — what the stock market currently thinks the whole company is worth.",
    "Beta": "A measure of how much a stock moves relative to the overall market. 1.0 = moves with the market, above 1.0 = more volatile, below 1.0 = less volatile.",
    "P/E Ratio": "Price-to-Earnings ratio: stock price ÷ earnings per share. Shows how many dollars investors pay today for each dollar of annual profit.",
    "EV/EBITDA": "Enterprise Value ÷ EBITDA — a valuation multiple comparing the value of the whole business to its cash operating profit, useful for comparing companies with different debt levels.",
    "EBIT": "Earnings Before Interest and Taxes — operating profit from the core business, before financing costs (interest) and taxes are subtracted, so businesses can be compared regardless of how they're financed.",
    "EBITDA": "Earnings Before Interest, Taxes, Depreciation & Amortization — EBIT with non-cash depreciation/amortization added back. A rough proxy for cash operating profit.",
    "FCFF": "Free Cash Flow to Firm — the actual cash a business generates each year that belongs to all capital providers (both debt and equity holders), after operating costs, taxes, and reinvestment.",
    "NOPAT": "Net Operating Profit After Tax — EBIT after subtracting the taxes the operating business would owe, ignoring any effect of debt or interest.",
    "D&A": "Depreciation & Amortization — a non-cash accounting expense that spreads the cost of long-lived assets over their useful life. Added back when calculating cash flow because no cash actually leaves the business.",
    "CapEx": "Capital Expenditure — actual cash spent buying or upgrading long-term assets (equipment, property, technology) needed to keep the business running and growing.",
    "Change in WC": "Change in Working Capital — the short-term cash tied up in day-to-day operations (inventory, receivables, payables). An increase means cash gets temporarily 'trapped' in the business.",
    "WACC": "Weighted Average Cost of Capital — the blended annual return required by everyone who financed the company (shareholders and lenders), weighted by how much of the capital each group provides. This is the discount rate used to bring future cash flows back to today's value.",
    "CAPM": "Capital Asset Pricing Model — a formula for estimating the return shareholders require: Risk-Free Rate + Beta × Equity Risk Premium.",
    "Risk-Free Rate": "The theoretical return on a zero-default-risk investment, approximated using the yield on 10-year U.S. Treasury bonds.",
    "Equity Risk Premium": "The extra return investors demand for holding stocks instead of a risk-free asset like government bonds, to compensate for the added risk.",
    "Cost of Debt": "The effective interest rate a company pays on its borrowings, typically estimated as interest expense ÷ average total debt.",
    "Terminal Value": "The estimated value of every cash flow a business will generate after the explicit forecast period ends, assuming it settles into stable, constant growth forever. Usually the largest single piece of the valuation.",
    "Terminal Growth Rate": "The assumed constant annual growth rate a business will sustain forever beyond the forecast window — usually kept modest (near long-run GDP growth) since no company can outgrow the economy indefinitely.",
    "Gordon Growth Model": "The formula used to calculate Terminal Value, assuming cash flow grows at a constant rate forever: TV = Final Year Cash Flow × (1 + g) ÷ (WACC − g).",
    "Enterprise Value": "The total value of a company's core business operations, belonging to all capital providers (debt + equity) combined — independent of how the company happens to be financed.",
    "Net Debt": "Total debt minus cash and cash equivalents — the debt burden a company would still carry if it used all its cash on hand to pay down debt.",
    "Equity Value": "The value that belongs specifically to shareholders: Enterprise Value minus Net Debt. Dividing by shares outstanding gives an intrinsic value per share.",
    "Intrinsic Value": "What an asset is calculated to actually be worth based on its underlying fundamentals (like a DCF), as opposed to whatever price the market currently happens to quote.",
}


def tip(key: str, label: str | None = None) -> str:
    """Return an inline hover-tooltip <abbr> span for a glossary term.

    Use only inside st.markdown(..., unsafe_allow_html=True) calls, never
    inside st.info/st.success/etc., which do not render raw HTML.
    """
    definition = TERMS.get(key, "").replace('"', "'")
    text = (label or key).replace('"', "'")
    return f'<abbr class="dcf-term" title="{definition}">{text}</abbr>'


def glossary_box(keys: list[str], title: str = "📖 Key Terms in This Section") -> None:
    """Render a collapsible glossary of the given TERMS keys as styled cards."""
    with st.expander(title):
        cards = "".join(
            f'<div class="term-card"><div class="term-name">{k}</div>'
            f'<div class="term-def">{TERMS.get(k, "")}</div></div>'
            for k in keys
        )
        st.markdown(cards, unsafe_allow_html=True)


def teach(markdown_text: str) -> None:
    """Render a bordered 'teaching note' callout between charts/tables."""
    with st.container(border=True):
        st.markdown(markdown_text, unsafe_allow_html=True)


st.markdown(
    """
    <style>
    abbr.dcf-term {
        text-decoration: underline dotted #2ECC71;
        text-decoration-thickness: 1.5px;
        cursor: help;
        font-weight: 600;
        color: #1a7a4c;
    }
    .term-card {
        background: #F5F7FA;
        border-left: 4px solid #2ECC71;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 10px;
    }
    .term-card .term-name { font-weight: 700; color: #2C3E50; font-size: 0.95rem; }
    .term-card .term-def { color: #5a6b7d; font-size: 0.85rem; margin-top: 3px; line-height: 1.4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached data-fetch wrappers
#
# st.cache_data avoids re-hitting the FMP/yfinance APIs on every widget
# interaction (e.g. moving a slider), which would otherwise burn through
# FMP's 250 requests/day free-tier limit almost instantly. Each wrapper is a
# thin pass-through to the corresponding src.data_fetcher function.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def cached_income_statements(symbol: str, api_key: str) -> pd.DataFrame:
    return fetcher.fetch_income_statements(symbol, api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_cash_flow_statements(symbol: str, api_key: str) -> pd.DataFrame:
    return fetcher.fetch_cash_flow_statements(symbol, api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_balance_sheets(symbol: str, api_key: str) -> pd.DataFrame:
    return fetcher.fetch_balance_sheets(symbol, api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_company_profile(symbol: str, api_key: str) -> dict:
    return fetcher.fetch_company_profile(symbol, api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_treasury_yield() -> float:
    return fetcher.fetch_treasury_yield()


@st.cache_data(ttl=900, show_spinner=False)
def cached_current_price(symbol: str) -> float:
    return fetcher.fetch_current_price(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_price_history(symbol: str) -> pd.DataFrame:
    return fetcher.fetch_price_history(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyst_targets(symbol: str) -> dict:
    return fetcher.fetch_analyst_targets(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_beta_fallback(symbol: str):
    return fetcher.fetch_beta_fallback(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_shares_outstanding(symbol: str, api_key: str) -> float:
    return fetcher.fetch_shares_outstanding(symbol, api_key)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "api_key" not in st.session_state:
    st.session_state.api_key = ENV_API_KEY
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "company_data" not in st.session_state:
    st.session_state.company_data = None


def load_company(symbol: str, api_key: str = "", demo: bool = False) -> None:
    """Run the full data-loading pipeline for `symbol` and populate session state.

    Shows a step-by-step status indicator ("Fetching financials..." ->
    "Computing WACC inputs..." -> "Done") so the user gets feedback during
    the several sequential API calls this requires. Any failure is caught
    and shown as a specific, actionable error message rather than crashing
    the app.
    """
    with st.status(f"Analysing {symbol}...", expanded=True) as status:
        try:
            if demo:
                status.write("Loading illustrative demo dataset (AAPL) — no API key required...")
                demo_data = fetcher.load_demo_data()
                income_df = demo_data["income_df"]
                cashflow_df = demo_data["cashflow_df"]
                balance_df = demo_data["balance_df"]
                profile = demo_data["profile"]
                symbol = "AAPL"
            else:
                status.write("Step 1/3 — Fetching income statement, cash flow statement & balance sheet from FMP...")
                income_df = cached_income_statements(symbol, api_key)
                cashflow_df = cached_cash_flow_statements(symbol, api_key)
                balance_df = cached_balance_sheets(symbol, api_key)
                status.write("Step 2/3 — Fetching company profile from FMP...")
                profile = cached_company_profile(symbol, api_key)

            status.write("Step 3/3 — Fetching market data (price, 10-year Treasury yield, analyst targets) via yfinance...")
            treasury_yield = cached_treasury_yield()

            try:
                current_price = cached_current_price(symbol)
            except fetcher.DataFetchError:
                current_price = profile.get("currentPrice") or 0.0

            try:
                price_history = cached_price_history(symbol)
            except fetcher.DataFetchError:
                price_history = None

            analyst = cached_analyst_targets(symbol)

            beta = profile.get("beta")
            if not beta:
                beta = cached_beta_fallback(symbol) or 1.0

            status.write("Computing historical FCFF, margins, and net debt...")
            fcff_df = processor.compute_fcff(income_df, cashflow_df, balance_df)
            historical_margins = processor.compute_historical_margins(income_df, cashflow_df)
            net_debt = processor.compute_net_debt(balance_df)

            if demo:
                # The hardcoded demo balance sheet still carries a shares
                # column; live FMP data does not (see fetch_shares_outstanding).
                shares = processor.get_latest_shares_outstanding(balance_df)
            else:
                try:
                    shares = cached_shares_outstanding(symbol, api_key)
                except fetcher.DataFetchError:
                    # Fall back to deriving share count from market cap and
                    # price if the enterprise-values endpoint is unavailable.
                    shares = (profile.get("marketCap") or 0) / current_price

            st.session_state.company_data = {
                "symbol": symbol,
                "income_df": income_df,
                "cashflow_df": cashflow_df,
                "balance_df": balance_df,
                "profile": profile,
                "treasury_yield": treasury_yield,
                "current_price": current_price,
                "price_history": price_history,
                "analyst": analyst,
                "beta": float(beta),
                "fcff_df": fcff_df,
                "historical_margins": historical_margins,
                "net_debt": net_debt,
                "shares": shares,
                "is_demo": demo,
            }
            st.session_state.data_loaded = True
            status.update(
                label=f"{profile.get('companyName', symbol)} loaded successfully.",
                state="complete",
            )
        except fetcher.DataFetchError as exc:
            status.update(label="Data fetch failed", state="error")
            st.error(str(exc))
            st.session_state.data_loaded = False
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected to the user
            status.update(label="Unexpected error", state="error")
            st.error(f"Unexpected error while loading {symbol}: {exc}")
            st.session_state.data_loaded = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💰 DCF Valuation Engine")
    st.caption("Institutional-grade DCF valuation, automated.")
    st.divider()

    if ENV_API_KEY:
        # A key is already available from the local .env file (gitignored,
        # never committed), so there's nothing for the user to type.
        st.caption("✅ FMP API key loaded from local environment (.env).")
    else:
        st.subheader("API Configuration")
        api_key_input = st.text_input(
            "FMP API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your free FMP API key from financialmodelingprep.com",
        )
        st.session_state.api_key = api_key_input
        st.markdown(
            "[Get your free key at financialmodelingprep.com]"
            "(https://financialmodelingprep.com/developer/docs/)"
        )

    st.divider()
    st.subheader("Company Search")
    ticker_input = st.text_input("Ticker", value="AAPL").strip().upper()

    col_analyse, col_demo = st.columns(2)
    with col_analyse:
        analyse_clicked = st.button("🔍 Analyse", use_container_width=True)
    with col_demo:
        demo_clicked = st.button("🎮 Demo Mode", use_container_width=True)

    if demo_clicked:
        load_company("AAPL", demo=True)
    elif analyse_clicked:
        if not st.session_state.api_key:
            st.error("Please enter an FMP API key above, or use Demo Mode.")
        elif not ticker_input:
            st.error("Please enter a ticker symbol.")
        else:
            load_company(ticker_input, api_key=st.session_state.api_key, demo=False)

    if st.session_state.data_loaded and st.session_state.company_data:
        data = st.session_state.company_data
        hist_margins = data["historical_margins"]

        st.divider()
        st.subheader("DCF Assumptions")

        projection_years = st.slider("Projection Years", 3, 10, 5)

        st.markdown("**Revenue Growth**")
        default_g13 = int(round(np.clip(hist_margins["revenue_growth_rate"], 0, 0.50) * 100))
        g13_pct = st.slider("Year 1-3 Growth Rate (%)", 0, 50, default_g13)
        growth_1_3 = g13_pct / 100

        default_g45 = int(round(np.clip(hist_margins["revenue_growth_rate"] * 0.7, 0, 0.30) * 100))
        g45_pct = st.slider("Year 4+ Growth Rate (%)", 0, 30, default_g45)
        growth_4_5 = g45_pct / 100

        st.markdown("**Profitability**")
        default_margin = int(round(np.clip(hist_margins["ebit_margin"], 0, 0.50) * 100))
        margin_pct = st.slider("EBIT Margin (%)", 0, 50, default_margin)
        ebit_margin = margin_pct / 100

        default_tax = int(round(np.clip(hist_margins["effective_tax_rate"], 0, 0.40) * 100))
        tax_pct = st.slider("Tax Rate (%)", 0, 40, default_tax)
        tax_rate = tax_pct / 100

        st.markdown("**Reinvestment**")
        default_capex = int(round(np.clip(hist_margins["capex_as_pct_revenue"], 0, 0.30) * 100))
        capex_pct_val = st.slider("CapEx as % Revenue", 0, 30, default_capex)
        capex_pct = capex_pct_val / 100

        default_da = int(round(np.clip(hist_margins["da_as_pct_revenue"], 0, 0.20) * 100))
        da_pct_val = st.slider("D&A as % Revenue", 0, 20, default_da)
        da_pct = da_pct_val / 100

        st.markdown("**Terminal Value**")
        terminal_growth_pct = st.slider("Terminal Growth Rate (%)", 0.0, 5.0, 2.5, 0.1)
        terminal_growth_rate = terminal_growth_pct / 100

        st.markdown("**WACC Components**")
        beta_input = st.number_input(
            "Beta", value=float(data["beta"]), min_value=0.0, max_value=3.0, step=0.01
        )
        erp_pct = st.slider("Equity Risk Premium (%)", 3.0, 8.0, 5.5, 0.1)
        erp = erp_pct / 100

        cost_of_equity = wacc_calc.compute_cost_of_equity(beta_input, data["treasury_yield"], erp)
        cost_of_debt = wacc_calc.compute_cost_of_debt(data["income_df"], data["balance_df"])
        capital_structure = wacc_calc.compute_capital_structure(data["profile"], data["balance_df"])
        wacc_value = wacc_calc.compute_wacc(
            cost_of_equity,
            cost_of_debt,
            capital_structure["equity_weight"],
            capital_structure["debt_weight"],
            tax_rate,
        )
        st.info(f"**Computed WACC: {wacc_value:.2%}**")

        st.button("🔄 Recalculate DCF", use_container_width=True)

        assumptions = {
            "base_revenue": float(data["income_df"].iloc[-1]["revenue"]),
            "projection_years": projection_years,
            "growth_rates": [growth_1_3] * min(3, projection_years)
            + [growth_4_5] * max(0, projection_years - 3),
            "ebit_margin": ebit_margin,
            "tax_rate": tax_rate,
            "capex_pct": capex_pct,
            "da_pct": da_pct,
            "terminal_growth_rate": terminal_growth_rate,
        }


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if not st.session_state.data_loaded or not st.session_state.company_data:
    st.title("💰 DCF Valuation Engine")
    st.markdown(
        "Build an institutional-grade Discounted Cash Flow valuation for any "
        "publicly listed company in seconds."
    )
    st.markdown(
        "**Get started:** enter your free [Financial Modeling Prep]"
        "(https://financialmodelingprep.com) API key and a ticker in the "
        "sidebar and click **Analyse** — or click **Demo Mode** to explore "
        "the tool instantly with sample Apple (AAPL) data, no API key required."
    )
    st.stop()

data = st.session_state.company_data
income_df = data["income_df"]
cashflow_df = data["cashflow_df"]
balance_df = data["balance_df"]
profile = data["profile"]
fcff_df = data["fcff_df"]
net_debt = data["net_debt"]
shares = data["shares"]
current_price = data["current_price"]
symbol = data["symbol"]
price_history = data["price_history"]
analyst = data["analyst"]

latest_income = income_df.iloc[-1]
eps = latest_income["netIncome"] / shares if shares else None
trailing_pe = current_price / eps if eps else None
enterprise_value_market = (profile.get("marketCap") or 0) + net_debt
ev_ebitda = (
    enterprise_value_market / latest_income["ebitda"]
    if latest_income.get("ebitda")
    else None
)

if data.get("is_demo"):
    st.warning(
        "📊 **Demo Mode** — showing illustrative, hardcoded AAPL financials "
        "(not a live feed). Enter a free FMP API key in the sidebar for "
        "real-time analysis of any ticker."
    )

result = dcf.run_full_dcf(
    assumptions, historical_data={}, wacc=wacc_value, shares=shares, net_debt=net_debt
)
upside_pct = (result["intrinsic_value_per_share"] - current_price) / current_price

st.title(f"{profile.get('companyName', symbol)} ({symbol})")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Company Overview",
        "💰 DCF Model",
        "📈 Sensitivity Analysis",
        "⚖️ Valuation Gauge",
        "📋 Assumptions & Methodology",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1 — Company Overview
# ---------------------------------------------------------------------------
with tab1:
    teach(
        "🧭 **Where we're starting.** Before building a valuation, we first need to know "
        "who this company is and how the market currently sees it. The five numbers below "
        "are the standard snapshot analysts pull up first — hover any underlined term for "
        f"a plain-English definition. {tip('Market Cap')}, {tip('Beta')}, {tip('P/E Ratio')} "
        f"and {tip('EV/EBITDA')} all come from today's market price, not our model — they're "
        "the baseline we'll compare our own DCF estimate against later."
    )

    st.subheader(profile.get("companyName", symbol))
    st.caption(
        f"{profile.get('sector', '—')} · {profile.get('industry', '—')} · "
        f"{profile.get('exchange', '—')}"
    )
    if profile.get("description"):
        with st.expander("Company Description"):
            st.write(profile["description"])

    c1, c2, c3, c4, c5 = st.columns(5)
    market_cap = profile.get("marketCap") or 0
    c1.metric("Market Cap", f"${market_cap / 1e9:,.1f}B")
    c2.metric("Current Price", f"${current_price:,.2f}")
    c3.metric("Beta", f"{data['beta']:.2f}")
    c4.metric("P/E Ratio", f"{trailing_pe:,.1f}x" if trailing_pe else "n/a")
    c5.metric("EV/EBITDA", f"{ev_ebitda:,.1f}x" if ev_ebitda else "n/a")

    glossary_box(["Market Cap", "Beta", "P/E Ratio", "EV/EBITDA", "EBITDA"])

    st.divider()
    st.markdown("#### 📈 Price History vs. Where the Market Thinks This Stock Is Worth")
    teach(
        "The chart below plots the actual traded stock price over time, alongside two "
        f"reference lines: this app's {tip('Intrinsic Value')} estimate (from the DCF Model "
        "tab) and the average price target published by professional equity analysts. "
        "Neither line is a prediction of where the price will go — they're both independent "
        "opinions of what the stock is *worth*, which the market price may or may not agree with."
    )

    if price_history is not None:
        st.plotly_chart(
            plot.plot_historical_price_vs_dcf(
                price_history,
                result["intrinsic_value_per_share"],
                analyst.get("targetMeanPrice"),
            ),
            use_container_width=True,
        )
    else:
        st.info("Historical price data unavailable for this ticker.")

    st.divider()
    st.markdown("#### 🏛️ The Foundation: Historical Revenue, EBIT & FCFF")
    teach(
        "Every DCF starts by looking backward before it looks forward. This chart shows "
        f"the last 5 fiscal years of actual, reported revenue, {tip('EBIT')} and {tip('FCFF')} — "
        "the real cash the business generated. These historical trends are what the sliders "
        "in the sidebar default to; the model assumes the *future* will resemble a reasonable "
        "extrapolation of the *past*, adjusted by whatever assumptions you choose to dial in."
    )

    st.plotly_chart(
        plot.plot_historical_financials(income_df, fcff_df), use_container_width=True
    )

    st.markdown("**Historical Margins & Ratios**")
    teach(
        f"This table breaks down, year by year, how fast revenue grew, what share of revenue "
        f"turned into {tip('EBIT', 'operating profit (EBIT)')}, and what share of profit was "
        "actually paid to the government in tax. These three ratios are exactly the levers "
        "you'll adjust with sliders in the sidebar to project the next few years — so it's "
        "worth noting whether they've been trending up, down, or holding steady."
    )
    margins_table = pd.DataFrame(
        {
            "Fiscal Year": income_df["date"].dt.year,
            "Revenue Growth": income_df["revenue"].pct_change().map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            ),
            "EBIT Margin": (income_df["ebit"] / income_df["revenue"]).map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            ),
            "Effective Tax Rate": income_df["effective_tax_rate"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            ),
        }
    )
    st.dataframe(margins_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2 — DCF Model
# ---------------------------------------------------------------------------
with tab2:
    teach(
        "🧮 **This is the model's final answer** — everything else on this tab shows the "
        "working behind it. Adjust any slider in the sidebar (growth, margins, WACC, "
        f"terminal growth) and this number recalculates instantly. It represents what one "
        f"share would be worth if you bought the entire future stream of {tip('FCFF')} the "
        "business is projected to generate, valued in today's dollars."
    )

    col_value, col_verdict = st.columns([2, 1])
    with col_value:
        st.metric(
            "DCF Intrinsic Value per Share",
            f"${result['intrinsic_value_per_share']:,.2f}",
            delta=f"{upside_pct:+.1%} vs current price ${current_price:,.2f}",
        )
    with col_verdict:
        if upside_pct > 0:
            st.success(f"🟢 Undervalued by {upside_pct:.1%}")
        else:
            st.error(f"🔴 Overvalued by {abs(upside_pct):.1%}")

    st.divider()
    st.markdown("#### 🌉 The Value Bridge: From Future Cash to Today's Dollars")
    teach(
        f"This chart shows how each piece adds up to {tip('Enterprise Value')}: the present "
        f"value of every projected year of {tip('FCFF')}, plus the present value of the "
        f"{tip('Terminal Value')} (the value of everything beyond the forecast window). "
        f"{tip('Net Debt')} is then subtracted to bridge down to what belongs to shareholders. "
        "Notice how much of the bar is terminal value — that's usually the majority of the "
        "total, which is why the Sensitivity Analysis tab exists."
    )

    st.plotly_chart(
        plot.plot_dcf_bridge(result["pv_fcffs"], result["pv_terminal_value"], net_debt),
        use_container_width=True,
    )

    st.divider()
    st.markdown("#### 📋 Year-by-Year Projection")
    teach(
        "Row by row, this is the actual forecast: revenue grown forward using the sidebar's "
        f"growth-rate sliders, converted into projected {tip('EBIT')} using the assumed EBIT "
        f"margin, then converted into {tip('FCFF')} — the cash figure that ultimately gets "
        "discounted back to today. Each year further out is worth less in present-value terms, "
        "even if the raw cash flow is bigger, because it's more distant and less certain."
    )

    start_year = int(income_df["date"].dt.year.iloc[-1]) + 1
    proj_table = pd.DataFrame(
        {
            "Fiscal Year": range(start_year, start_year + assumptions["projection_years"]),
            "Revenue": result["projected_revenues"],
            "EBIT (assumed margin)": [r * ebit_margin for r in result["projected_revenues"]],
            "FCFF": result["projected_fcffs"],
            "PV(FCFF)": result["pv_fcffs"],
        }
    )
    st.dataframe(
        proj_table.style.format(
            {
                "Revenue": "${:,.0f}",
                "EBIT (assumed margin)": "${:,.0f}",
                "FCFF": "${:,.0f}",
                "PV(FCFF)": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(
        plot.plot_revenue_projection(
            income_df["date"], income_df["revenue"], result["projected_revenues"]
        ),
        use_container_width=True,
    )

    st.divider()
    st.markdown("#### 🧱 Building One Year of FCFF from the Ground Up")
    teach(
        f"Pick a projected year below to see exactly how its {tip('FCFF')} figure is built, "
        f"piece by piece: start with {tip('NOPAT')} (profit after tax, ignoring financing), "
        f"add back {tip('D&A')} (a non-cash expense), then subtract {tip('CapEx')} (real cash "
        f"spent on equipment/property) and {tip('Change in WC')}. What's left is the actual "
        "cash that could be paid out to investors that year."
    )

    year_choice = st.selectbox(
        "Select year for FCFF build-up",
        options=list(range(1, assumptions["projection_years"] + 1)),
        format_func=lambda y: f"Year {y} (FY{start_year + y - 1})",
    )
    rev_y = result["projected_revenues"][year_choice - 1]
    fcff_components = {
        "NOPAT": rev_y * ebit_margin * (1 - tax_rate),
        "D&A": rev_y * da_pct,
        "CapEx": rev_y * capex_pct,
        "Change in WC": rev_y * 0.01,
    }
    st.plotly_chart(
        plot.plot_fcff_waterfall(
            fcff_components, f"Year {year_choice} (FY{start_year + year_choice - 1})"
        ),
        use_container_width=True,
    )

    st.divider()
    st.markdown("#### 🥧 Explicit Forecast vs. Terminal Value")
    teach(
        f"This pie shows what fraction of today's {tip('Enterprise Value')} comes from cash "
        f"flows we explicitly forecasted, versus the {tip('Terminal Value')} — the catch-all "
        "for everything beyond the forecast window. A large terminal-value slice isn't a flaw; "
        "it's simply how long-lived businesses work. It does mean the valuation leans heavily "
        "on one long-term growth assumption, which is exactly what the next tab stress-tests."
    )

    st.plotly_chart(
        plot.plot_pv_breakdown_pie(result["pv_fcffs"], result["pv_terminal_value"]),
        use_container_width=True,
    )

    st.divider()
    st.markdown("### ⚖️ WACC Breakdown")
    teach(
        f"{tip('WACC')} is the single most important number in this whole model — it's the "
        "annual discount rate applied to every future dollar. It blends two things: how much "
        f"return shareholders demand ({tip('CAPM')}), and how much it costs the company to "
        f"borrow ({tip('Cost of Debt')}, reduced because interest is tax-deductible), weighted "
        "by how much of the company is funded by each. A higher WACC means future cash is "
        "discounted more aggressively, which lowers the valuation — and vice versa."
    )
    w1, w2, w3 = st.columns(3)
    w1.metric("Cost of Equity (Re)", f"{cost_of_equity:.2%}")
    w2.metric("Cost of Debt, after-tax", f"{cost_of_debt * (1 - tax_rate):.2%}")
    w3.metric("WACC", f"{wacc_value:.2%}")
    st.caption(
        "Cost of Equity (CAPM) = Risk-free rate + Beta × Equity Risk Premium — the "
        "annual return shareholders require. Cost of Debt = Interest Expense ÷ "
        "Average Total Debt, then taxed at the effective tax rate because interest "
        "payments are tax-deductible. WACC blends both, weighted by the market "
        "value of equity and book value of debt — it is the single discount rate "
        "applied to every cash flow (and the terminal value) in this model."
    )

    glossary_box(
        [
            "EBIT",
            "FCFF",
            "NOPAT",
            "D&A",
            "CapEx",
            "Change in WC",
            "Terminal Value",
            "Enterprise Value",
            "Net Debt",
            "WACC",
            "CAPM",
            "Risk-Free Rate",
            "Equity Risk Premium",
            "Cost of Debt",
        ]
    )

# ---------------------------------------------------------------------------
# Tab 3 — Sensitivity Analysis
# ---------------------------------------------------------------------------
with tab3:
    teach(
        "🎯 **Why this tab exists.** The single 'base case' number on the DCF Model tab is "
        "just *one* point estimate drawn from a whole range of plausible outcomes — every "
        f"input (growth, margin, {tip('WACC')}, {tip('Terminal Growth Rate')}) is an assumption, "
        "not a fact. Instead of trusting one number, professional analysts always ask: *'how "
        "much does the answer change if I'm wrong about my assumptions?'* The grids below "
        "answer exactly that, by recalculating the full valuation across a whole range of "
        "input combinations at once."
    )

    st.markdown("#### 🌡️ Heatmap 1 — WACC vs. Terminal Growth Rate")
    teach(
        f"These are the two assumptions the valuation is most sensitive to, since they "
        f"directly drive the {tip('Terminal Value')} (usually the majority of total value). "
        "Each cell recalculates the full intrinsic value per share for that WACC/growth "
        "combination — greener cells suggest the stock looks more undervalued at that "
        "assumption pair, redder cells suggest more overvalued, relative to today's price."
    )

    wacc_range = np.round(
        np.arange(max(0.02, wacc_value - 0.03), wacc_value + 0.035, 0.005), 4
    )
    growth_range = np.round(np.arange(0.01, 0.045, 0.005), 4)
    wacc_growth_df = sens.wacc_vs_growth_sensitivity(
        assumptions, wacc_range, growth_range, {}, shares, net_debt
    )
    st.plotly_chart(
        plot.plot_sensitivity_heatmap(
            wacc_growth_df, "Intrinsic Value: WACC vs Terminal Growth Rate", current_price
        ),
        use_container_width=True,
    )

    st.markdown("#### 🌡️ Heatmap 2 — EBIT Margin vs. Revenue Growth")
    teach(
        f"This second grid stress-tests the two assumptions driving the *explicit* forecast "
        f"years instead: how profitable each future dollar of revenue is ({tip('EBIT')} "
        "margin) and how fast revenue actually grows. Together with the WACC/growth grid "
        "above, this shows which of your assumptions the final answer actually hinges on."
    )

    margin_range = np.round(
        np.arange(max(0.0, ebit_margin - 0.10), ebit_margin + 0.12, 0.02), 4
    )
    rev_growth_range = np.round(np.arange(0.0, 0.28, 0.03), 4)
    margin_growth_df = sens.margin_vs_growth_sensitivity(
        assumptions, margin_range, rev_growth_range, wacc_value, {}, shares, net_debt
    )
    st.plotly_chart(
        plot.plot_sensitivity_heatmap(
            margin_growth_df, "Intrinsic Value: EBIT Margin vs Revenue Growth", current_price
        ),
        use_container_width=True,
    )

    st.markdown("#### 🐂🐻 Bull, Base & Bear Scenarios")
    teach(
        "Rather than varying two assumptions in isolation, this chart bundles *every* "
        "assumption together into three coherent stories: an optimistic case (Bull), the "
        "assumptions currently set in the sidebar (Base), and a pessimistic case (Bear). "
        "This mirrors how analysts actually present a valuation range in practice — as a "
        "spread of outcomes, not a single confident number."
    )

    scenarios = sens.scenario_analysis(assumptions, wacc_value, {}, shares, net_debt)
    st.plotly_chart(
        plot.plot_scenario_comparison(scenarios, current_price), use_container_width=True
    )

    tv_pct = result["tv_as_pct_of_ev"]
    sensitivity_level = (
        "highly" if tv_pct > 0.8 else "moderately" if tv_pct > 0.6 else "relatively less"
    )
    st.info(
        f"**Key Insight:** {tv_pct:.0%} of Enterprise Value comes from the terminal "
        f"value, meaning this valuation is **{sensitivity_level} sensitive** to the "
        "terminal growth rate and WACC assumptions rather than the explicitly "
        "projected near-term cash flows."
    )

    glossary_box(["WACC", "Terminal Growth Rate", "Terminal Value", "EBIT", "Gordon Growth Model"])

# ---------------------------------------------------------------------------
# Tab 4 — Valuation Gauge
# ---------------------------------------------------------------------------
with tab4:
    teach(
        "🏁 **Putting it all in context.** A single DCF estimate means little on its own — "
        "it only becomes useful compared against other viewpoints on the same stock. The "
        f"gauge below shows where today's price sits relative to this model's {tip('Intrinsic Value')}; "
        "the table underneath adds two more independent reference points: what professional "
        "analysts currently expect, and where the price has actually traded over the past year."
    )

    st.plotly_chart(
        plot.plot_valuation_gauge(result["intrinsic_value_per_share"], current_price),
        use_container_width=True,
    )

    if price_history is not None:
        week52_low = price_history["Close"].tail(252).min()
        week52_high = price_history["Close"].tail(252).max()
        week52_range = f"${week52_low:,.2f} – ${week52_high:,.2f}"
    else:
        week52_range = "n/a"

    target_mean = analyst.get("targetMeanPrice")

    st.markdown("#### 🧾 Four Independent Viewpoints, Side by Side")
    teach(
        "None of these four numbers are 'the truth' — each answers a slightly different "
        "question. Our DCF estimate is a bottom-up cash-flow view; the analyst consensus "
        "is the average opinion of professionals who cover this stock; the 52-week range "
        f"shows the market's own recent mood swings; and {tip('P/E Ratio')} is a quick "
        "relative-valuation sanity check against the company's own earnings."
    )

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Our DCF Estimate",
                "Analyst Consensus Target",
                "52-Week Range",
                "Trailing P/E",
            ],
            "Value": [
                f"${result['intrinsic_value_per_share']:,.2f}",
                f"${target_mean:,.2f}" if target_mean else "n/a",
                week52_range,
                f"{trailing_pe:,.1f}x" if trailing_pe else "n/a",
            ],
        }
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    direction = "undervalued" if upside_pct > 0 else "overvalued"
    # Dollar signs are escaped (\$) because st.markdown treats a pair of "$"
    # as inline LaTeX math - unescaped, "$293.02 ... $88.14" would be
    # rendered as a single garbled math expression instead of plain text.
    st.markdown(
        f"""
### Verdict
At the current price of **\\${current_price:,.2f}**, **{profile.get('companyName', symbol)}**
appears **{direction} by {abs(upside_pct):.1%}** relative to our DCF estimate of
**\\${result['intrinsic_value_per_share']:,.2f}** under base case assumptions.

However, this estimate rests heavily on the terminal growth rate and WACC
assumptions (see the Sensitivity Analysis tab — {tv_pct:.0%} of Enterprise
Value comes from the terminal value), and it does not capture qualitative
factors such as competitive dynamics, management execution, or near-term
market sentiment. Treat it as one well-reasoned estimate among a range of
plausible outcomes, not a precise price target.
        """
    )

    glossary_box(["Intrinsic Value", "Enterprise Value", "Equity Value", "P/E Ratio"])

# ---------------------------------------------------------------------------
# Tab 5 — Assumptions & Methodology
# ---------------------------------------------------------------------------
with tab5:
    teach(
        "📚 **The full reference.** This tab is the complete recipe and glossary for "
        "everything you've seen across the other four tabs — every formula, every "
        "assumption currently in effect, and every term used anywhere in this app."
    )

    st.markdown(
        """
### How this DCF works

1. **Historical financials** — the last 5 fiscal years of revenue, EBIT,
   free cash flow, debt and cash are pulled from filed financial statements.
2. **Projection** — revenue is grown forward for the projection window using
   an assumed growth rate, then converted into Free Cash Flow to Firm (FCFF)
   using assumed EBIT margin, tax rate, CapEx and D&A ratios.
3. **Terminal value** — after the projection window, the business is assumed
   to grow at a constant, modest rate forever (the Gordon Growth Model),
   capturing all value beyond the explicit forecast.
4. **WACC** — a blended discount rate is computed from the cost of equity
   (via CAPM) and the after-tax cost of debt, weighted by the market value
   of equity and book value of debt.
5. **Discounting** — every projected cash flow and the terminal value are
   discounted back to today's dollars using WACC.
6. **Equity bridge** — Enterprise Value (the whole business) minus net debt
   equals Equity Value; dividing by shares outstanding gives an intrinsic
   value per share, which is compared against the current market price.
        """
    )

    st.markdown("### Current Model Assumptions")
    assumptions_table = pd.DataFrame(
        {
            "Assumption": [
                "Projection Years",
                "Year 1-3 Revenue Growth",
                "Year 4+ Revenue Growth",
                "EBIT Margin",
                "Tax Rate",
                "CapEx % of Revenue",
                "D&A % of Revenue",
                "Terminal Growth Rate",
                "Beta",
                "Risk-Free Rate (10Y Treasury)",
                "Equity Risk Premium",
                "Cost of Equity",
                "Cost of Debt (pre-tax)",
                "WACC",
            ],
            "Value": [
                assumptions["projection_years"],
                f"{growth_1_3:.1%}",
                f"{growth_4_5:.1%}",
                f"{ebit_margin:.1%}",
                f"{tax_rate:.1%}",
                f"{capex_pct:.1%}",
                f"{da_pct:.1%}",
                f"{terminal_growth_rate:.1%}",
                f"{beta_input:.2f}",
                f"{data['treasury_yield']:.2%}",
                f"{erp:.1%}",
                f"{cost_of_equity:.2%}",
                f"{cost_of_debt:.2%}",
                f"{wacc_value:.2%}",
            ],
        }
    )
    st.dataframe(assumptions_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
### Data Sources & Limitations
- **Financial Modeling Prep (FMP)** — historical income statements, cash flow
  statements, balance sheets, and company profile. Free tier is limited to
  250 requests/day and may lag the very latest quarterly filing.
- **yfinance** — current price, 2-year price history, 10-year Treasury yield
  (risk-free rate), and analyst consensus targets. Unofficial and
  best-effort; occasionally rate-limited or incomplete for smaller tickers.

### Caveats
- DCF valuations are **highly sensitive** to the terminal growth rate and
  WACC — small changes in either can swing the result by 20-30%.
- Historical margins are a starting point, not a guarantee — they may not
  reflect future competitive, regulatory, or macroeconomic conditions.
- This model does **not** capture qualitative factors: management quality,
  product pipeline, litigation risk, or forward guidance.

**Disclaimer:** This is an educational tool built for portfolio purposes. It
is **not financial advice** and should not be used as the sole basis for any
investment decision.
        """
    )

    st.divider()
    glossary_box(list(TERMS.keys()), title="📖 Full Glossary — Every Term Used in This App")
