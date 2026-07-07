# DCF Valuation Engine

An interactive Discounted Cash Flow (DCF) valuation tool built as a Streamlit web app. It pulls real financial data for any publicly listed company, builds a DCF model automatically, and outputs an intrinsic value per share. Every assumption (growth, margins, WACC, terminal growth) is adjustable, and the valuation, sensitivity tables, and charts update instantly.

Built as a portfolio project for Quantitative Analyst / Financial Technology roles.

## Live Demo

[nakul-dcf-engine.streamlit.app](https://nakul-dcf-engine.streamlit.app/), includes a Demo Mode with no API key required.

## Key Features

- Pulls the last 5 years of income statement, cash flow statement, and balance sheet data for any ticker
- Derives historical Free Cash Flow to Firm (FCFF) and key operating margins
- Projects revenue and cash flow forward using adjustable assumptions (growth, margins, CapEx, D&A, tax rate)
- Computes a terminal value and full Weighted Average Cost of Capital (WACC), including CAPM cost of equity
- Discounts every projected cash flow and the terminal value back to present value, bridging Enterprise Value to an intrinsic value per share
- Compares the result against current market price, analyst consensus targets, and 52-week range
- Runs a full sensitivity analysis: WACC vs terminal growth, EBIT margin vs revenue growth, and Bull/Base/Bear scenarios
- Ships with a Demo Mode that loads a sample Apple (AAPL) dataset with no API key required

## Methodology

1. Historical Financials: fetch 5 years of revenue, EBIT, EBITDA, net income, operating cash flow, CapEx, D&A, debt, cash, and shares outstanding.
2. Project Future Cash Flows: grow revenue forward and apply assumed EBIT margin, tax rate, CapEx %, and D&A % to derive projected FCFF for each future year.
3. Terminal Value: assume the business grows at a modest, constant rate forever after the projection window (Gordon Growth Model). This typically represents 60-80% of total value.
4. WACC (Discount Rate): blend the cost of equity (via CAPM: risk-free rate + beta x equity risk premium) and the after-tax cost of debt, weighted by market value of equity and book value of debt.
5. Discount to Present Value: bring every projected cash flow and the terminal value back to today's dollars using WACC.
6. Bridge to Equity Value: subtract net debt from Enterprise Value to get Equity Value, then divide by shares outstanding to get intrinsic value per share, compared against current market price.

## Repo Structure

```
├── app.py                          # Streamlit application entry point
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # App theme configuration
└── src/
    ├── data_fetcher.py             # Pulls financials from FMP + market data from yfinance
    ├── financial_processor.py      # Derives historical FCFF and operating margins
    ├── dcf_model.py                # Cash flow projection, terminal value, valuation bridge
    ├── wacc_calculator.py          # CAPM cost of equity + WACC
    ├── sensitivity_analysis.py     # WACC/terminal growth and margin/growth grids, scenarios
    └── plotting.py                 # All chart functions
```

## Installation & Running Locally

```bash
git clone https://github.com/NakulPillai2407/dcf-valuation-engine.git
cd dcf-valuation-engine

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501), enter your FMP API key and a ticker in the sidebar, or click Demo Mode to explore immediately with no key.

Getting a free FMP API key: go to financialmodelingprep.com, create a free account, and copy your API key from the developer dashboard. Paste it into the sidebar of this app; it's stored only in your browser session, never hardcoded or persisted. (FMP also offers a 30% student discount on paid plans, if you outgrow the free tier.)

## Tech Stack

- Python, Streamlit, Plotly, Pandas / NumPy
- Financial Modeling Prep (FMP): historical income statements, cash flow statements, balance sheets, and company profile data. Free tier: 250 requests/day.
- yfinance: current price, price history, risk-free rate, and analyst consensus price targets. No API key required.

## Key Concepts

- DCF: estimates what a company is worth today by forecasting the cash it will generate in the future and discounting those cash flows back to present-day value, since a dollar received in five years is worth less than a dollar today.
- FCFF (Free Cash Flow to Firm): the cash a business generates that belongs to all capital providers, both debt and equity holders, before any financing decisions. It starts from EBIT rather than net income so the valuation is independent of how the company happens to be financed.
- WACC (Weighted Average Cost of Capital): the blended annual return required by shareholders and lenders, weighted by how much of the company's capital each group provides. It's the discount rate applied to every future cash flow in the model, and the most sensitive input in any DCF.
- Terminal Value: captures the value of everything the business generates after the explicit forecast window, assuming it settles into stable, perpetual growth. Typically 60-80% of total Enterprise Value.

## Limitations

DCF valuations are highly sensitive to the terminal growth rate and WACC assumptions: small changes in either can swing the result by 20-30%. Historical margins are a starting point, not a guarantee of future performance, and this model does not capture qualitative factors such as management quality, competitive dynamics, or forward guidance.

This is an educational tool built for portfolio purposes. It is not financial advice and should not be used as the sole basis for any investment decision.

## Author

**Nakul Pillai**
BSc Economics & Data Science, University of Southampton · Incoming MSc Financial Technology, Imperial College London

- LinkedIn: [linkedin.com/in/nakul-pillai](https://www.linkedin.com/in/nakul-pillai)
- GitHub: [@NakulPillai2407](https://github.com/NakulPillai2407)
