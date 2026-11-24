# Invest like the Swiss National Bank

QARM II project, HEC Lausanne.

A family office in Geneva wants to follow the US equity portfolio of the SNB, but with only 30 to 50 stocks.
Every quarter we read the SNB holdings in its 13F filing and we build the portfolio of n stocks with the
smallest tracking error vs the SNB (long only, fully invested, max weight per stock, sector weights within
2% of the SNB, turnover limit). The client can exclude some sectors and can give us a tracking error budget
to add our own views (momentum) with Black-Litterman, using the SNB portfolio as the equilibrium portfolio.

## Files

- `data_preparation.py` : cleans the raw data and saves it in `data/processed` (run it once)
- `functions.py` : covariance matrix (Ledoit-Wolf), choice of the stocks, optimization, Black-Litterman, backtest
- `app.py` : the Streamlit app

## Run the app

```
pip install -r requirements.txt
python3 data_preparation.py
python3 -m streamlit run app.py
```

## Data

Put these files in `data/raw` (names must be exactly these ones):

| File | Source | Columns |
|---|---|---|
| `crsp_daily.csv` | WRDS, CRSP Stock v2, Daily Stock File | PERMNO, CUSIP, HdrCUSIP, DlyCalDt, DlyPrc, DlyRet, DlyCap, SecurityType, SecuritySubType, ShareType, PrimaryExch, Ticker, SecurityNm |
| `ccm_sectors.csv` | WRDS, CRSP/Compustat Merged, Fundamentals Annual | LPERMNO, datadate, gsector, ggroup, gind, gsubind |
| `snb_13f.csv` | WRDS, LSEG (Thomson Reuters) 13F holdings (s34), manager Swiss National Bank (mgrno 13631) | rdate, fdate, mgrname, cusip, shares |
| `ff_factors_daily.csv` | Kenneth French website, Fama/French 3 Factors daily (CSV) | used for the risk-free rate |

The CRSP data starts 2 years before the first rebalancing because we need 2 years of returns for the
covariance matrix. The CRSP data is under the WRDS licence, so the data files (raw and processed) are not on GitHub.
