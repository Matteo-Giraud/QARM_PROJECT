# We run this file once to clean the raw data and save it in data/processed.
# The app then only reads the processed files.

import pandas as pd
import numpy as np
import os

# Raw files (WRDS at the CEDIF + Kenneth French website)
CRSP_FILE = 'data/raw/crsp_daily.csv'         # CRSP Stock v2 - Daily Stock File
SECTOR_FILE = 'data/raw/ccm_sectors.csv'      # CRSP/Compustat Merged v2 - Fundamentals Annual
F13_FILE = 'data/raw/snb_13f.csv'             # 13F holdings of the SNB
FF_FILE = 'data/raw/ff_factors_daily.csv'     # Fama/French 3 factors daily (for the risk-free rate)

UNIVERSE_SIZE = 500     # the 500 largest US stocks, close to the S&P 500
FILING_LAG = 45         # a 13F is public 45 days after the end of the quarter
FIRST_REBALANCING = '2015-01-01'
EST_WINDOW = 504        # 2 years of data needed before the first rebalancing
MIN_OBS = 0.8           # a stock needs at least 80% of returns in these 2 years

# common shares listed on the NYSE, AMEX or NASDAQ (CRSP v2 codes)
FILTERS = {'securitytype': ['EQTY'], 'securitysubtype': ['COM'], 'sharetype': ['NS'], 'primaryexch': ['N', 'A', 'Q']}

os.makedirs('data/processed', exist_ok = True)



# 1. 13F of the SNB

f13 = pd.read_csv(F13_FILE, dtype = str)
f13.columns = f13.columns.str.lower()

# the file can contain other managers, we keep the SNB only
if 'mgrname' in f13.columns:
    f13 = f13[f13['mgrname'].str.upper().str.contains('SWISS NATIONAL', na = False)]
f13['rdate'] = pd.to_datetime(f13['rdate'])
f13['shares'] = pd.to_numeric(f13['shares'], errors = 'coerce')
f13['cusip'] = f13['cusip'].str[:8]          # 8 first characters to match CRSP

# when a 13F was amended there are 2 filings for the same quarter : we keep the last one (fdate = filing date)
f13['fdate'] = pd.to_datetime(f13['fdate'])
f13 = f13[f13['fdate'] == f13.groupby('rdate')['fdate'].transform('max')]
report_dates = sorted(f13['rdate'].unique())
print('Number of 13F :', len(report_dates))

# we will need the prices and market caps at the end of each quarter
quarter_end_days = []
for d in report_dates:
    quarter_end_days += list(pd.date_range(d - pd.Timedelta(days = 7), d))



# 2. CRSP daily file

# The file is very big so we read it in chunks. We keep :
#   - the daily returns of every stock
#   - the price, market cap and CUSIP at the end of each quarter
#   - the last ticker and name of each stock
ret_chunks, snap_chunks, name_chunks = [], [], []

for chunk in pd.read_csv(CRSP_FILE, chunksize = 1000000, dtype = str):
    chunk.columns = chunk.columns.str.lower()

    for col, values in FILTERS.items():
        chunk = chunk[chunk[col].isin(values)]

    chunk['date'] = pd.to_datetime(chunk['dlycaldt'])
    chunk['permno'] = chunk['permno'].astype(int)
    chunk['dlyret'] = pd.to_numeric(chunk['dlyret'], errors = 'coerce')
    chunk['dlyprc'] = pd.to_numeric(chunk['dlyprc'], errors = 'coerce').abs()
    chunk['dlycap'] = pd.to_numeric(chunk['dlycap'], errors = 'coerce')

    ret_chunks.append(chunk[['permno', 'date', 'dlyret']])
    snap_chunks.append(chunk.loc[chunk['date'].isin(quarter_end_days), ['permno', 'date', 'cusip', 'dlyprc', 'dlycap']])
    name_chunks.append(chunk.groupby('permno')[['ticker', 'securitynm']].last())

crsp = pd.concat(ret_chunks)
snapshots = pd.concat(snap_chunks)
names = pd.concat(name_chunks).groupby(level = 0).last()

# matrix of daily returns : dates in rows, stocks in columns
returns = crsp.pivot_table(values = 'dlyret', index = 'date', columns = 'permno')
print('Daily returns :', returns.shape)
days = returns.index



# 3. Sectors (GICS codes from Compustat)

sectors = pd.read_csv(SECTOR_FILE)
sectors.columns = sectors.columns.str.lower()
sectors['datadate'] = pd.to_datetime(sectors['datadate'])
sectors = sectors.dropna(subset = ['lpermno', 'gsector']).sort_values('datadate')
sectors['lpermno'] = sectors['lpermno'].astype(int)
for col in ['gsector', 'ggroup', 'gind', 'gsubind']:
    # codes as text without decimals, e.g. 45.0 -> '45' (a missing code becomes '0')
    sectors[col] = pd.to_numeric(sectors[col], errors = 'coerce').fillna(0).astype(int).astype(str)


def sectors_at(date, permnos):
    # GICS codes of the last fiscal year before the date
    s = sectors[sectors['datadate'] <= date].groupby('lpermno').last()
    s = s.reindex(permnos)
    s['gsector'] = s['gsector'].fillna('99')       # unknown sector
    return s[['gsector', 'ggroup', 'gind', 'gsubind']].fillna('')



# 4. One portfolio of the SNB per quarter

quarters = []

for rdate in report_dates:
    # we rebalance on the first trading day 45 days after the end of the quarter
    pos = days.searchsorted(rdate + pd.Timedelta(days = FILING_LAG))
    if pos >= len(days) - 1 or days[pos] < pd.Timestamp(FIRST_REBALANCING) or pos < EST_WINDOW:
        continue
    rebalance_date = days[pos]

    # prices and market caps on the last trading day of the quarter
    snap = snapshots[(snapshots['date'] <= rdate) & (snapshots['date'] > rdate - pd.Timedelta(days = 7))]
    snap = snap.sort_values('date').groupby('permno').last()

    # universe = the 500 largest market caps
    universe = snap['dlycap'].nlargest(UNIVERSE_SIZE).index

    # we keep the stocks with enough returns in the 2 years before the rebalancing
    window = returns.iloc[pos - EST_WINDOW:pos][universe]
    universe = universe[window.notna().mean().values >= MIN_OBS]

    # SNB holdings : CUSIP -> permno, value = number of shares x price
    holdings = f13[f13['rdate'] == rdate].copy()
    cusip_to_permno = snap.reset_index().drop_duplicates('cusip').set_index('cusip')['permno']
    holdings['permno'] = holdings['cusip'].map(cusip_to_permno)
    matched = holdings['permno'].notna().mean()
    holdings = holdings.dropna(subset = ['permno'])
    holdings['permno'] = holdings['permno'].astype(int)
    holdings['value'] = holdings['shares'] * snap.loc[holdings['permno'], 'dlyprc'].values
    snb_value = holdings.groupby('permno')['value'].sum()
    in_universe = snb_value.reindex(universe).fillna(0).sum() / snb_value.sum()
    snb_value = snb_value.reindex(universe).fillna(0)

    quarters.append({'report_date': rdate,
                     'rebalance_date': rebalance_date,
                     'permnos': np.array(universe),
                     'w_snb': (snb_value / snb_value.sum()).values,
                     'w_mkt': (snap.loc[universe, 'dlycap'] / snap.loc[universe, 'dlycap'].sum()).values,
                     'sectors': sectors_at(rdate, universe),
                     'nb_snb_stocks': len(holdings)})
    print(rdate.date(), '-> rebalancing', rebalance_date.date(), '| universe', len(universe),
          '| 13F matched with CRSP : {:.0%}'.format(matched), '| SNB value in the universe : {:.0%}'.format(in_universe))



# 5. Risk-free rate

market = pd.DataFrame(index = days)

# Kenneth French file : a few lines of text, then date (YYYYMMDD), Mkt-RF, SMB, HML, RF in %
ff = pd.read_csv(FF_FILE, header = None, skiprows = 1, names = ['date', 'mkt_rf', 'smb', 'hml', 'rf'],
                 dtype = str, on_bad_lines = 'skip')
ff = ff[ff['date'].str.strip().str.match(r'^\d{8}$', na = False)]
ff['date'] = pd.to_datetime(ff['date'].str.strip(), format = '%Y%m%d')
market['rf'] = (ff.set_index('date')['rf'].astype(float) / 100).reindex(days).ffill().fillna(0)



# 6. Save

all_permnos = np.unique(np.concatenate([q['permnos'] for q in quarters]))
returns[all_permnos].to_pickle('data/processed/returns.pkl')
names.reindex(all_permnos).to_pickle('data/processed/names.pkl')
market.to_pickle('data/processed/market.pkl')
pd.to_pickle(quarters, 'data/processed/quarters.pkl')
print('Saved', len(quarters), 'quarters and', len(all_permnos), 'stocks')