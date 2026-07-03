# DCF Valuation Engine

An interactive Discounted Cash Flow (DCF) valuation tool, built as a Streamlit web app, that pulls real financial data for any publicly listed company, builds a complete DCF model automatically, and presents the results the way an equity research analyst or investment banking associate would. Adjust every assumption — growth, margins, WACC, terminal growth — and watch the valuation, sensitivity tables, and charts update instantly.

## Live Demo

*https://nakul-dcf-engine.streamlit.app/*

## What It Does

- Pulls the last 5 years of income statement, cash flow statement, and balance sheet data for any ticker
- Automatically derives historical Free Cash Flow to Firm (FCFF) and key operating margins
- Projects revenue and cash flow forward using fully adjustable assumptions (growth, margins, CapEx, D&A, tax rate)
- Computes a terminal value and a full Weighted Average Cost of Capital (WACC), including CAPM cost of equity
- Discounts every projected cash flow and the terminal value back to today, bridging Enterprise Value down to an intrinsic value per share
- Compares the result against the current market price, analyst consensus targets, and 52-week range
- Runs a full sensitivity analysis — WACC vs terminal growth, EBIT margin vs revenue growth, and Bull/Base/Bear scenarios
- Ships with a **Demo Mode** that loads a sample Apple (AAPL) dataset with no API key required

## The Methodology — 6 Steps

1. **Historical Financials** — fetch 5 years of revenue, EBIT, EBITDA, net income, operating cash flow, CapEx, D&A, debt, cash, and shares outstanding.
2. **Project Future Cash Flows** — grow revenue forward and apply assumed EBIT margin, tax rate, CapEx %, and D&A % to derive projected FCFF for each future year.
3. **Terminal Value** — assume the business grows at a modest, constant rate forever after the projection window (Gordon Growth Model), capturing all value beyond the explicit forecast — typically 60-80% of total value.
4. **WACC (Discount Rate)** — blend the cost of equity (via CAPM: risk-free rate + beta × equity risk premium) and the after-tax cost of debt, weighted by market value of equity and book value of debt.
5. **Discount to Present Value** — bring every projected cash flow and the terminal value back to today's dollars using WACC.
6. **Bridge to Equity Value** — subtract net debt from Enterprise Value to get Equity Value, then divide by shares outstanding to get an intrinsic value per share, compared against the current market price.

## Data Sources

- **[Financial Modeling Prep (FMP)](https://financialmodelingprep.com)** — historical income statements, cash flow statements, balance sheets, and company profile (market cap, beta, sector). Free tier: 250 requests/day.
- **[yfinance](https://github.com/ranaroussi/yfinance)** — current price, 2-year price history, the 10-year US Treasury yield (risk-free rate), and analyst consensus price targets. No API key required.

## Getting Your Free FMP API Key

1. Go to [financialmodelingprep.com](https://financialmodelingprep.com) and create a free account.
2. Copy your API key from the developer dashboard.
3. Paste it into the sidebar of this app (stored only in your browser session — never hardcoded or persisted).

*(FMP also offers a 30% student discount on paid plans, if you outgrow the free tier.)*

## Installation & Running Locally

```bash
# Clone the repository
git clone <this-repo-url>
cd dcf-valuation-engine

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`), enter your FMP API key and a ticker in the sidebar, or click **Demo Mode** to explore immediately with no key.

## Screenshots

*(placeholder — add screenshots of the Company Overview, DCF Model, Sensitivity Analysis, and Valuation Gauge tabs here)*

## Key Financial Concepts

**What is a DCF?**
A Discounted Cash Flow valuation estimates what a company is worth today by forecasting all the cash it will generate in the future and discounting those future cash flows back to present-day value — because a dollar received in five years is worth less than a dollar today.

**What is FCFF?**
Free Cash Flow to Firm is the cash a business generates that belongs to *all* capital providers — both debt and equity holders — before any financing decisions. It starts from EBIT (not net income) specifically so the valuation is independent of how the company happens to be financed.

**What is WACC?**
The Weighted Average Cost of Capital is the blended annual return required by everyone who has financed the company — shareholders and lenders — weighted by how much of the company's capital each group provides. It's the discount rate applied to every future cash flow in the model, and the single most sensitive input in any DCF.

**What is Terminal Value?**
Since cash flows can't be explicitly projected forever, the terminal value captures the value of everything the business generates after the explicit forecast window, assuming it settles into stable, perpetual growth. It typically represents 60-80% of total Enterprise Value.

## Limitations & Disclaimer

DCF valuations are highly sensitive to the terminal growth rate and WACC assumptions — small changes in either can swing the result by 20-30%. Historical margins are a starting point, not a guarantee of future performance, and this model does not capture qualitative factors such as management quality, competitive dynamics, or forward guidance.

**This is an educational tool built for portfolio purposes. It is not financial advice and should not be used as the sole basis for any investment decision.**

## Technologies Used

- Python
- Streamlit
- Plotly
- Financial Modeling Prep API
- yfinance
- Pandas / NumPy

## Author

**Nakul Pillai**
MSc Financial Technology, Imperial College London

- LinkedIn: *(add link)*
- GitHub: *(add link)*
