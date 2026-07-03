"""
data_fetcher.py
================
All external data retrieval for the DCF Valuation Engine.

Two data sources are combined:
  1. Financial Modeling Prep (FMP)  -> historical fundamentals (income statement,
     cash flow statement, balance sheet, company profile). Requires a free API key.
  2. yfinance                       -> live/near-live market data (current price,
     historical price series, the 10-year Treasury yield used as the risk-free
     rate, and analyst price targets). No API key required.

Every public function here validates its inputs, wraps network calls in
try/except, and raises a `DataFetchError` with a message that is informative
enough to show directly in the Streamlit UI. Nothing in this module talks to
Streamlit directly (no `st.` calls) - the app layer decides how to display
errors and loading states.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

import pandas as pd
import requests
import yfinance as yf

# FMP retired its legacy "/api/v3" endpoints for any key issued after
# 31 Aug 2025 - all requests now go through the "/stable" API, which takes
# the ticker as a "symbol" query parameter instead of a path segment.
FMP_BASE_URL = "https://financialmodelingprep.com/stable"
REQUEST_TIMEOUT_SECONDS = 15


class DataFetchError(Exception):
    """Raised when a data source cannot supply the data we need.

    We use a dedicated exception (rather than letting raw requests/yfinance
    exceptions bubble up) so the Streamlit layer can catch one predictable
    error type and show a clean, user-facing message.
    """


def _get_fmp_json(endpoint: str, params: Dict[str, Any]) -> Any:
    """Low-level GET helper shared by all FMP-backed fetch functions.

    Parameters
    ----------
    endpoint : the FMP path segment, e.g. "income-statement/AAPL"
    params   : query string parameters, must include "apikey"

    Returns the parsed JSON body. Raises DataFetchError on any failure:
    network issues, non-200 status, rate limiting, or an FMP error payload
    (FMP returns HTTP 200 with an "Error Message" field on bad requests,
    which requests.raise_for_status() would not catch on its own).
    """
    url = f"{FMP_BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        raise DataFetchError(
            f"Network error while contacting Financial Modeling Prep: {exc}. "
            "Check your internet connection and try again."
        ) from exc

    # FMP returns a JSON body - often with a specific "Error Message" - even
    # on error status codes, so we try to surface that exact reason before
    # falling back to a generic message per status code.
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict) and "Error Message" in payload:
        raise DataFetchError(f"FMP error: {payload['Error Message']}")

    if response.status_code == 401:
        raise DataFetchError(
            "FMP rejected the API key (401 Unauthorized). Double-check the key "
            "you entered in the sidebar, or generate a new one at "
            "financialmodelingprep.com."
        )
    if response.status_code == 403:
        raise DataFetchError(
            "FMP returned 403 Forbidden. This endpoint may require a paid plan, "
            "or your free-tier daily request limit (250/day) has been reached."
        )
    if response.status_code == 429:
        raise DataFetchError(
            "FMP rate limit hit (429 Too Many Requests). Please wait ~60 seconds "
            "before trying again."
        )
    if response.status_code != 200:
        raise DataFetchError(
            f"FMP request failed with HTTP {response.status_code} for endpoint "
            f"'{endpoint}'. The ticker may be invalid or the endpoint unavailable "
            "on your plan."
        )

    if not payload:
        raise DataFetchError(
            f"FMP returned no data for endpoint '{endpoint}'. The ticker symbol "
            "may be incorrect or this data may not be covered by the free tier."
        )

    return payload


def fetch_income_statements(symbol: str, api_key: str, years: int = 5) -> pd.DataFrame:
    """Fetch the last `years` of annual income statements from FMP.

    The income statement shows a company's PROFITABILITY over a period: how
    much revenue it earned, what it cost to earn that revenue, and what was
    left over as profit at various stages (gross profit, operating profit,
    net income).

    Fields extracted and their financial meaning:
      - revenue              : total sales generated in the period
      - grossProfit          : revenue minus cost of goods sold
      - ebitda               : earnings before interest, tax, depreciation &
                                amortisation - a proxy for cash operating profit
      - ebit (operatingIncome): earnings before interest & tax - the profit
                                the business generates from operations alone,
                                before financing costs or taxes. This is the
                                key input to FCFF because it is capital-structure
                                neutral (it doesn't matter how much debt the
                                company has).
      - netIncome            : the "bottom line" profit after interest and tax
      - incomeTaxExpense     : tax actually charged in the period
      - interestExpense      : cost of servicing debt
      - effective_tax_rate   : incomeTaxExpense / pretaxIncome - the actual
                                (not statutory) tax rate the company pays,
                                used to convert EBIT into NOPAT

    Returns
    -------
    pd.DataFrame sorted from oldest to newest (ascending by date), one row
    per fiscal year, with the columns named above.

    Limitations: FMP occasionally omits `interestExpense` or reports it as a
    negative number depending on the filing; both cases are normalised here
    to a positive magnitude.
    """
    if not symbol:
        raise DataFetchError("No ticker symbol provided.")
    if not api_key:
        raise DataFetchError("No FMP API key provided. Enter one in the sidebar.")

    data = _get_fmp_json(
        "income-statement",
        {"symbol": symbol.upper(), "limit": years, "apikey": api_key},
    )

    rows = []
    for item in data:
        pretax_income = item.get("incomeBeforeTax")
        tax_expense = item.get("incomeTaxExpense") or 0
        # Guard against division by zero / missing pretax income.
        effective_tax_rate = (
            tax_expense / pretax_income
            if pretax_income not in (None, 0)
            else None
        )
        rows.append(
            {
                "date": item.get("date"),
                "revenue": item.get("revenue"),
                "grossProfit": item.get("grossProfit"),
                "ebitda": item.get("ebitda"),
                # FMP's "ebit" field is Earnings Before Interest & Tax
                # directly; "operatingIncome" is a close but distinct line
                # item (it can exclude some non-operating items included in
                # EBIT), so we prefer "ebit" and fall back only if missing.
                "ebit": item.get("ebit") if item.get("ebit") is not None else item.get("operatingIncome"),
                "netIncome": item.get("netIncome"),
                "incomeTaxExpense": tax_expense,
                "interestExpense": abs(item.get("interestExpense") or 0),
                "effective_tax_rate": effective_tax_rate,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise DataFetchError(f"No income statement data available for '{symbol}'.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_cash_flow_statements(symbol: str, api_key: str, years: int = 5) -> pd.DataFrame:
    """Fetch the last `years` of annual cash flow statements from FMP.

    The cash flow statement reconciles NET INCOME (an accounting figure) with
    actual CASH movements. Net income includes non-cash charges (like
    depreciation) and is distorted by working-capital timing (e.g. a sale
    recorded as revenue before the cash is collected). The cash flow
    statement strips these distortions out, which is exactly what we need to
    build Free Cash Flow to Firm.

    Fields extracted:
      - operatingCashFlow          : cash generated by core business operations
      - capitalExpenditure         : cash spent on long-term assets (property,
                                      equipment, etc). FMP reports this as a
                                      negative number; we store it as a
                                      positive magnitude for clarity.
      - depreciationAndAmortization: non-cash expense that reduced accounting
                                      profit but did not consume cash - added
                                      back when computing FCFF
      - changeInWorkingCapital     : cash tied up (or freed) by changes in
                                      receivables, payables and inventory

    Returns
    -------
    pd.DataFrame sorted oldest to newest.
    """
    if not symbol:
        raise DataFetchError("No ticker symbol provided.")
    if not api_key:
        raise DataFetchError("No FMP API key provided. Enter one in the sidebar.")

    data = _get_fmp_json(
        "cash-flow-statement",
        {"symbol": symbol.upper(), "limit": years, "apikey": api_key},
    )

    rows = []
    for item in data:
        rows.append(
            {
                "date": item.get("date"),
                "operatingCashFlow": item.get("operatingCashFlow"),
                "capitalExpenditure": abs(item.get("capitalExpenditure") or 0),
                "depreciationAndAmortization": item.get("depreciationAndAmortization"),
                "changeInWorkingCapital": item.get("changeInWorkingCapital"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise DataFetchError(f"No cash flow statement data available for '{symbol}'.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_balance_sheets(symbol: str, api_key: str, years: int = 5) -> pd.DataFrame:
    """Fetch the last `years` of annual balance sheets from FMP.

    The balance sheet is a snapshot (not a period flow like the other two
    statements) of what the company OWNS (assets) and OWES (liabilities),
    with the difference belonging to shareholders (equity):

        Assets = Liabilities + Shareholders' Equity

    Fields extracted:
      - totalDebt                : short-term + long-term interest-bearing debt
      - cashAndCashEquivalents    : cash and near-cash assets, which can be
                                    used to pay down debt - this is why we net
                                    it against debt later ("net debt")
      - totalStockholdersEquity  : book value of equity (accounting value,
                                    NOT what we use for WACC - we use market
                                    cap for that instead, see wacc_calculator.py)
      - shortTermDebt / longTermDebt : debt maturing within vs beyond 12 months

    Note: shares outstanding is NOT on this statement (FMP's balance sheet
    schema has no reliable share-count field - "commonStock" is the par
    value of issued stock in dollars, not a share count). Use
    `fetch_shares_outstanding` for that figure instead.

    Returns
    -------
    pd.DataFrame sorted oldest to newest.
    """
    if not symbol:
        raise DataFetchError("No ticker symbol provided.")
    if not api_key:
        raise DataFetchError("No FMP API key provided. Enter one in the sidebar.")

    data = _get_fmp_json(
        "balance-sheet-statement",
        {"symbol": symbol.upper(), "limit": years, "apikey": api_key},
    )

    rows = []
    for item in data:
        short_term_debt = item.get("shortTermDebt") or 0
        long_term_debt = item.get("longTermDebt") or 0
        total_debt = item.get("totalDebt")
        if total_debt is None:
            total_debt = short_term_debt + long_term_debt

        rows.append(
            {
                "date": item.get("date"),
                "totalDebt": total_debt,
                "cashAndCashEquivalents": item.get("cashAndCashEquivalents") or 0,
                "totalStockholdersEquity": item.get("totalStockholdersEquity"),
                "shortTermDebt": short_term_debt,
                "longTermDebt": long_term_debt,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise DataFetchError(f"No balance sheet data available for '{symbol}'.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_shares_outstanding(symbol: str, api_key: str) -> float:
    """Fetch the most recent shares outstanding figure via FMP's enterprise-values endpoint.

    FMP's balance sheet statement no longer carries a share-count field, so
    shares outstanding is sourced separately here (the enterprise-values
    endpoint reports "numberOfShares" alongside each period's stock price
    and market cap). This is the divisor used to convert total Equity Value
    into an intrinsic value per share.
    """
    if not symbol:
        raise DataFetchError("No ticker symbol provided.")
    if not api_key:
        raise DataFetchError("No FMP API key provided. Enter one in the sidebar.")

    data = _get_fmp_json(
        "enterprise-values",
        {"symbol": symbol.upper(), "limit": 1, "apikey": api_key},
    )
    shares = data[0].get("numberOfShares")
    if not shares or shares <= 0:
        raise DataFetchError(f"No shares outstanding figure available for '{symbol}'.")
    return float(shares)


def fetch_company_profile(symbol: str, api_key: str) -> Dict[str, Any]:
    """Fetch descriptive/market data for a company from FMP's profile endpoint.

    Returns a dict with:
      - companyName, sector, industry, description, exchange, currency
      - marketCap    : current market value of equity (E in WACC = share
                        price * shares outstanding). This reflects what
                        investors are willing to pay TODAY, unlike book equity
                        on the balance sheet which reflects historical cost.
      - beta         : measures how much the stock's returns move relative to
                        the overall market. beta = 1 means the stock tends to
                        move in line with the market; beta = 1.5 means it
                        tends to move 50% more than the market in either
                        direction (more volatile / more systematic risk);
                        beta < 1 means it's less volatile than the market.
                        Beta feeds directly into the cost of equity via CAPM.
      - currentPrice : latest traded price per FMP (yfinance is used as the
                        primary source for price elsewhere, but this is a
                        useful cross-check)
    """
    if not symbol:
        raise DataFetchError("No ticker symbol provided.")
    if not api_key:
        raise DataFetchError("No FMP API key provided. Enter one in the sidebar.")

    data = _get_fmp_json("profile", {"symbol": symbol.upper(), "apikey": api_key})

    if not isinstance(data, list) or not data:
        raise DataFetchError(f"No company profile found for '{symbol}'.")

    item = data[0]
    return {
        "companyName": item.get("companyName"),
        "sector": item.get("sector"),
        "industry": item.get("industry"),
        "marketCap": item.get("marketCap"),
        "beta": item.get("beta"),
        "currentPrice": item.get("price"),
        "description": item.get("description"),
        "exchange": item.get("exchange"),
        "currency": item.get("currency", "USD"),
    }


def fetch_treasury_yield() -> float:
    """Fetch the current 10-year US Treasury yield via yfinance ("^TNX").

    The 10-year Treasury yield is the standard proxy for the RISK-FREE RATE
    (Rf) in the CAPM formula. It is considered "risk-free" because the US
    government is (in practice) assumed not to default on its own domestic
    currency debt, and the 10-year tenor roughly matches the long-term
    horizon of an equity investment / DCF projection.

    yfinance's "^TNX" ticker quotes the yield directly in percentage points
    (e.g. a quote of 4.47 means a 4.47% yield), so we divide by 100 to get a
    decimal fraction (e.g. 0.0447).

    Returns the yield as a decimal (e.g. 0.045 for 4.5%). Falls back to a
    conservative default of 4.0% if the fetch fails, so the rest of the
    model can still run.
    """
    try:
        ticker = yf.Ticker("^TNX")
        history = ticker.history(period="5d")
        if history.empty:
            raise DataFetchError("^TNX price history is empty.")
        latest_quote = history["Close"].iloc[-1]
        return float(latest_quote) / 100.0
    except Exception:
        # Non-fatal: fall back to a reasonable long-run average rather than
        # blocking the whole DCF because of a transient yfinance hiccup.
        return 0.04


def fetch_current_price(symbol: str) -> float:
    """Fetch the latest closing price for `symbol` via yfinance.

    Used as the market price we compare our DCF intrinsic value against to
    determine whether the stock looks under- or over-valued.
    """
    if not symbol:
        raise DataFetchError("No ticker symbol provided.")
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="5d")
        if history.empty:
            raise DataFetchError(f"No recent price history found for '{symbol}'.")
        return float(history["Close"].iloc[-1])
    except DataFetchError:
        raise
    except Exception as exc:
        raise DataFetchError(
            f"Could not fetch the current price for '{symbol}' via yfinance: {exc}"
        ) from exc


def fetch_price_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical daily closing prices for `symbol` via yfinance.

    Used to plot the stock's price trajectory alongside our DCF intrinsic
    value and the analyst consensus target, giving visual context for how
    the current price compares to recent history.
    """
    if not symbol:
        raise DataFetchError("No ticker symbol provided.")
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)
        if history.empty:
            raise DataFetchError(f"No price history found for '{symbol}'.")
        return history.reset_index()[["Date", "Close"]]
    except DataFetchError:
        raise
    except Exception as exc:
        raise DataFetchError(
            f"Could not fetch price history for '{symbol}' via yfinance: {exc}"
        ) from exc


def fetch_analyst_targets(symbol: str) -> Dict[str, Optional[float]]:
    """Fetch analyst consensus price target and recommendation via yfinance.

    This is used purely as a SANITY CHECK alongside our own DCF estimate -
    it is Wall Street's aggregate near-term view, whereas a DCF is a
    long-run, assumption-driven intrinsic value estimate. The two can and
    often do disagree; that disagreement is itself informative.

    Returns a dict with keys: targetMeanPrice, targetHighPrice,
    targetLowPrice, recommendationKey. Any field that yfinance does not
    provide is returned as None rather than raising, since analyst target
    data availability is inconsistent across tickers.
    """
    result = {
        "targetMeanPrice": None,
        "targetHighPrice": None,
        "targetLowPrice": None,
        "recommendationKey": None,
    }
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        result["targetMeanPrice"] = info.get("targetMeanPrice")
        result["targetHighPrice"] = info.get("targetHighPrice")
        result["targetLowPrice"] = info.get("targetLowPrice")
        result["recommendationKey"] = info.get("recommendationKey")
    except Exception:
        # Non-fatal: analyst targets are a "nice to have" comparison, not a
        # required input to the DCF itself.
        pass
    return result


def load_demo_data() -> Dict[str, Any]:
    """Return a hardcoded, offline AAPL dataset for "Demo Mode".

    This lets a visitor (e.g. a recruiter) exercise the full app without
    signing up for an FMP API key. The figures below are approximate,
    illustrative AAPL fundamentals (FY2020-FY2024, USD) - close enough to
    real filed results to produce a realistic-looking DCF, but they are NOT
    guaranteed to match FMP's live data exactly and should never be treated
    as investment-grade figures. The UI must label this clearly as "Demo
    Mode - illustrative data" wherever it is used.

    Returns a dict with keys: income_df, cashflow_df, balance_df, profile -
    matching exactly the shapes returned by fetch_income_statements,
    fetch_cash_flow_statements, fetch_balance_sheets and
    fetch_company_profile respectively, so downstream code (financial
    processing, WACC, DCF) needs no special-casing for demo data.
    """
    dates = pd.to_datetime(
        ["2020-09-30", "2021-09-30", "2022-09-30", "2023-09-30", "2024-09-30"]
    )

    income_df = pd.DataFrame(
        {
            "date": dates,
            "revenue": [274515e6, 365817e6, 394328e6, 383285e6, 391035e6],
            "grossProfit": [104956e6, 152836e6, 170782e6, 169148e6, 180683e6],
            "ebitda": [81020e6, 123136e6, 130541e6, 125820e6, 134661e6],
            "ebit": [66288e6, 108949e6, 119437e6, 114301e6, 123216e6],
            "netIncome": [57411e6, 94680e6, 99803e6, 96995e6, 93736e6],
            "incomeTaxExpense": [9680e6, 14527e6, 19300e6, 16741e6, 29749e6],
            "interestExpense": [2873e6, 2645e6, 2931e6, 3933e6, 3932e6],
        }
    )
    income_df["effective_tax_rate"] = income_df["incomeTaxExpense"] / (
        income_df["netIncome"] + income_df["incomeTaxExpense"]
    )

    cashflow_df = pd.DataFrame(
        {
            "date": dates,
            "operatingCashFlow": [80674e6, 104038e6, 122151e6, 110543e6, 118254e6],
            "capitalExpenditure": [7309e6, 11085e6, 10708e6, 10959e6, 9447e6],
            "depreciationAndAmortization": [11056e6, 11284e6, 11104e6, 11519e6, 11445e6],
            "changeInWorkingCapital": [-1391e6, 4911e6, 1200e6, -1930e6, 3651e6],
        }
    )

    balance_df = pd.DataFrame(
        {
            "date": dates,
            "totalDebt": [112436e6, 124719e6, 120069e6, 111088e6, 106629e6],
            "cashAndCashEquivalents": [38016e6, 34940e6, 23646e6, 29965e6, 29943e6],
            "totalStockholdersEquity": [65339e6, 63090e6, 50672e6, 62146e6, 56950e6],
            "sharesOutstanding": [
                16976763000,
                16426786000,
                15943425000,
                15550061000,
                15116786000,
            ],
            "shortTermDebt": [13769e6, 15613e6, 21110e6, 15807e6, 22511e6],
            "longTermDebt": [98667e6, 109106e6, 98959e6, 95281e6, 84118e6],
        }
    )

    profile = {
        "companyName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3400000000000.0,
        "beta": 1.24,
        "currentPrice": 225.0,
        "description": (
            "Apple Inc. designs, manufactures, and markets smartphones, personal "
            "computers, tablets, wearables, and accessories, and sells a variety "
            "of related services. (Illustrative demo profile.)"
        ),
        "exchange": "NASDAQ",
        "currency": "USD",
    }

    return {
        "income_df": income_df,
        "cashflow_df": cashflow_df,
        "balance_df": balance_df,
        "profile": profile,
    }


def fetch_beta_fallback(symbol: str) -> Optional[float]:
    """Fetch beta via yfinance as a fallback if FMP does not return one.

    Beta measures a stock's volatility relative to the broader market and is
    a required input to CAPM / cost of equity. If FMP's profile endpoint
    returns None (which happens for some smaller or foreign tickers on the
    free tier), we try yfinance's `info` dict as a second source before
    giving up and asking the user to enter it manually.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        beta = info.get("beta")
        return float(beta) if beta is not None else None
    except Exception:
        return None
