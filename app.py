import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

import functions as f

st.set_page_config(page_title = 'Invest like the SNB', layout = 'wide')

COLORS = {'Portfolio': '#1f3b73', 'SNB': '#d52b1e', 'Market': '#8c8c8c'}


# Data

# cache_resource : the data and the covariance matrices are computed only once, not at each click
@st.cache_resource
def load_data():
    data = {'returns': pd.read_pickle('data/processed/returns.pkl'),
            'quarters': pd.read_pickle('data/processed/quarters.pkl'),
            'market': pd.read_pickle('data/processed/market.pkl'),
            'names': pd.read_pickle('data/processed/names.pkl')}
    return data


@st.cache_resource
def get_covariances(shrinkage):
    data = load_data()
    return f.covariance_matrices(data['returns'], data['quarters'], shrinkage)


# The backtest takes a few seconds, so the results are saved on the disk (persist = 'disk') and every
# setting we already tried is instant, even after restarting the app. data_version changes when we run
# data_preparation.py again, so we never see old results.
@st.cache_data(persist = 'disk')
def run_backtest(n_stocks, te_budget, exclusions, w_max, turnover_max, cost_bp, sector_tol, shrinkage, data_version):
    cov_matrices, deltas = get_covariances(shrinkage)
    return f.backtest(load_data(), cov_matrices, n_stocks, te_budget, exclusions, w_max, turnover_max, cost_bp,
                      sector_tol)


@st.cache_data(persist = 'disk')
def get_te_curve(exclusions, w_max, sector_tol, shrinkage, data_version):
    cov_matrices, deltas = get_covariances(shrinkage)
    k = len(cov_matrices) - 1
    return f.te_curve(load_data(), cov_matrices, k, [10, 20, 30, 40, 50, 75, 100], exclusions, w_max, sector_tol)


if not os.path.exists('data/processed/quarters.pkl'):
    st.error('No data found : run "python3 data_preparation.py" first (see README).')
    st.stop()

data = load_data()
data_version = os.path.getmtime('data/processed/quarters.pkl')



# Sidebar : questions to the client

st.sidebar.title('Your preferences')

n_stocks = st.sidebar.select_slider('How many stocks do you want to hold ?', options = [20, 30, 40, 50], value = 40)

te_budget = st.sidebar.select_slider('How far can we go from the SNB to add our views ?',
                                     options = [0, 0.01, 0.02, 0.03, 0.05], value = 0,
                                     format_func = lambda x: 'Copy the SNB' if x == 0 else '{:.0%}'.format(x),
                                     help = 'Maximum tracking error vs the SNB, the copy included. With few stocks the '
                                            'copy already has some tracking error (see the cost of simplicity), so we add '
                                            'our momentum views (Black-Litterman) only if your budget is above it.')

exclusions = st.sidebar.multiselect('Sectors you do not want to invest in', list(f.EXCLUSIONS))
exclusions = tuple(sorted(exclusions))

with st.sidebar.expander('Other settings'):
    w_max = st.select_slider('Maximum weight per stock', options = [x / 100 for x in range(5, 21)], value = 0.10,
                             format_func = lambda x: '{:.0%}'.format(x))
    turnover_options = {'10%': 0.10, '20%': 0.20, '30%': 0.30, '50%': 0.50, 'No limit': None}
    turnover_max = turnover_options[st.radio('Maximum turnover per quarter', list(turnover_options),
                                             index = 2, horizontal = True)]
    tolerance_options = {'1%': 0.01, '2%': 0.02, '3%': 0.03, '5%': 0.05, 'No limit': 1.}
    sector_tol = tolerance_options[st.radio('Maximum sector deviation from the SNB', list(tolerance_options),
                                            index = 1, horizontal = True)]
    cost_bp = st.number_input('Trading costs (bp of the amount traded)', 0, 100, 10)
    shrinkage = st.checkbox('Ledoit-Wolf shrinkage of the covariance matrix', value = True)

with st.spinner('Running the backtest for your settings (can take up to a minute the first time)...'):
    daily, results, last = run_backtest(n_stocks, te_budget, exclusions, w_max, turnover_max, cost_bp, sector_tol,
                                        shrinkage, data_version)

cov_matrices, deltas = get_covariances(shrinkage)
q = data['quarters'][-1]
perf = f.performance(daily)
start, end = daily.index[0], daily.index[-1]
nb_years = (end - start).days / 365.25

# CRSP names look like "APPLE INC; COM NONE; CONS" : we keep the name before the first ';'.
# If a company has 2 share classes in the universe (e.g. Berkshire A and B), we add the class
names = data['names'].reindex(q['permnos'])['securitynm']
company = names.str.split(';').str[0]
twice = company.duplicated(keep = False)
company[twice] = company[twice] + ' (' + names[twice].str.split(';').str[1].str.strip() + ')'

# table with everything we know about the stocks at the last rebalancing
table = pd.DataFrame({'Ticker': data['names'].reindex(q['permnos'])['ticker'].values,
                      'Company': company.values,
                      'Sector': q['sectors']['gsector'].map(f.SECTOR_NAMES).values,
                      'Portfolio': last['weights'].values,
                      'SNB': last['w_snb'].values,
                      'Market': last['w_mkt'].values,
                      'Excluded': last['excluded']}, index = q['permnos'])
table['Difference'] = table['Portfolio'] - table['SNB']



# Header and recommendation

st.title('Invest like the Swiss National Bank')
st.caption('A portfolio of a few US stocks that follows the US equity portfolio of the SNB (13F filings), '
           'with your own constraints.')

col1, col2, col3, col4 = st.columns(4)
col1.metric('Annual return', '{:.1%}'.format(perf.loc['Portfolio', 'Annual return']),
            '{:+.2%} vs SNB'.format(perf.loc['Portfolio', 'Annual return'] - perf.loc['SNB', 'Annual return']))
col2.metric('Tracking error vs SNB', '{:.2%}'.format(perf.loc['Portfolio', 'Tracking error vs SNB']))
col3.metric('Information ratio', '{:.2f}'.format(perf.loc['Portfolio', 'Information ratio']))
col4.metric('Trading costs per year', '{:.2%}'.format(results['cost'].sum() / nb_years))

te_copy = results['te_copy'].iloc[-1]
text = ('**Our recommendation :** with **' + str(n_stocks) + ' stocks**, the closest copy of the SNB has an '
        'expected tracking error of **{:.1%}** per year.'.format(te_copy))
if len(exclusions) > 0:
    text += ' This includes your exclusions (' + ', '.join(exclusions).lower() + ').'
if te_budget > 0 and te_budget > te_copy:
    text += ' With your budget of {:.0%} we add our momentum views on top of the copy.'.format(te_budget)
if te_budget > 0 and te_budget <= te_copy:
    text += (' Your budget of {:.0%} is already used by the copy, so there is no room for our views '
             '(you need more than {:.1%}).'.format(te_budget, te_copy))
text += (' From ' + str(start.year) + ' to ' + str(end.year) + ' this portfolio made **{:.1%}** per year '
         'against {:.1%} for the SNB, after trading costs.'.format(perf.loc['Portfolio', 'Annual return'],
                                                                     perf.loc['SNB', 'Annual return']))
st.info(text)

tab1, tab2, tab3 = st.tabs(['SNB portfolio', 'Our portfolio', 'Backtest'])



# Tab 1 : what does the SNB hold ?

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric('US stocks in the last 13F', q['nb_snb_stocks'])
    col1.caption('Quarter ending ' + q['report_date'].strftime('%d.%m.%Y') + ', public on '
                 + q['rebalance_date'].strftime('%d.%m.%Y'))
    col2.metric('Weight of the 10 largest positions', '{:.1%}'.format(table['SNB'].nlargest(10).sum()))
    col3.metric('Expected tracking error SNB vs market', '{:.2%}'.format(results['te_snb_market'].iloc[-1]),
                help = 'Market = the same 500 stocks weighted by market cap.')

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader('Largest positions')
        top = table.sort_values('SNB', ascending = False).head(15)
        top = top[['Ticker', 'Company', 'Sector', 'SNB', 'Market']]
        st.dataframe(top.style.format({'SNB': '{:.2%}', 'Market': '{:.2%}'}), hide_index = True)
    with right:
        st.subheader('Sectors : SNB vs market')
        sector_weights = table.groupby('Sector')[['SNB', 'Market']].sum().sort_values('SNB')
        fig = go.Figure()
        fig.add_bar(y = sector_weights.index, x = sector_weights['SNB'], name = 'SNB', orientation = 'h',
                    marker_color = COLORS['SNB'])
        fig.add_bar(y = sector_weights.index, x = sector_weights['Market'], name = 'Market', orientation = 'h',
                    marker_color = COLORS['Market'])
        fig.update_layout(barmode = 'group', height = 420, xaxis_tickformat = '.0%', margin = dict(t = 10, b = 0),
                          legend = dict(orientation = 'h', y = 1.12))
        st.plotly_chart(fig)

    st.subheader('Is the SNB just the market ?')
    left, right = st.columns([1.3, 1])
    with left:
        fig = px.line(results, x = 'rebalance_date', y = 'te_snb_market',
                      labels = {'rebalance_date': '', 'te_snb_market': 'Expected tracking error vs market'})
        fig.update_traces(line_color = COLORS['SNB'])
        fig.update_layout(height = 300, yaxis_tickformat = '.2%', margin = dict(t = 10, b = 0))
        st.plotly_chart(fig)
        st.caption('If the SNB was exactly the market, this tracking error would be 0.')
    with right:
        not_held = table[(table['SNB'] == 0)].sort_values('Market', ascending = False)
        st.write('**Large US stocks the SNB does not hold : ' + str(len(not_held)) + '**')
        st.dataframe(not_held.head(10)[['Ticker', 'Company', 'Sector', 'Market']].style.format({'Market': '{:.2%}'}),
                     hide_index = True)



# Tab 2 : the portfolio we propose

with tab2:
    held = table[table['Portfolio'] > 0.00001].sort_values('Portfolio', ascending = False)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Stocks held', len(held))
    col2.metric('Expected tracking error', '{:.2%}'.format(results['te'].iloc[-1]))
    col3.metric('Active share vs SNB', '{:.1%}'.format(0.5 * table['Difference'].abs().sum()))
    col4.metric('Turnover last quarter', '{:.1%}'.format(results['turnover'].iloc[-1]))
    if results['note'].iloc[-1] != '':
        st.warning('Last rebalancing : ' + results['note'].iloc[-1] + '.')

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader('Portfolio')
        st.dataframe(held[['Ticker', 'Company', 'Sector', 'Portfolio', 'SNB', 'Difference']]
                     .style.format({'Portfolio': '{:.2%}', 'SNB': '{:.2%}', 'Difference': '{:+.2%}'}),
                     hide_index = True, height = 420)
    with right:
        st.subheader('Sectors : portfolio vs SNB')
        sector_weights = table.groupby('Sector')[['Portfolio', 'SNB']].sum()
        snb_name = 'SNB'
        if len(exclusions) > 0:
            # the sector constraints use the SNB without the excluded stocks, rescaled to 100%
            snb_allowed = table['SNB'].where(~table['Excluded'], 0)
            sector_weights['SNB'] = (snb_allowed / snb_allowed.sum()).groupby(table['Sector']).sum()
            snb_name = 'SNB without your exclusions'
        sector_weights = sector_weights.sort_values('SNB')
        fig = go.Figure()
        fig.add_bar(y = sector_weights.index, x = sector_weights['Portfolio'], name = 'Portfolio', orientation = 'h',
                    marker_color = COLORS['Portfolio'])
        fig.add_bar(y = sector_weights.index, x = sector_weights['SNB'], name = snb_name, orientation = 'h',
                    marker_color = COLORS['SNB'])
        fig.update_layout(barmode = 'group', height = 420, xaxis_tickformat = '.0%', margin = dict(t = 10, b = 0),
                          legend = dict(orientation = 'h', y = 1.12))
        st.plotly_chart(fig)
        if sector_tol < 1:
            st.caption('Our sector weights stay within +/- {:.0%} of the '.format(sector_tol) + snb_name + '.')
        if len(exclusions) > 0:
            excluded = table[table['Excluded'] & (table['SNB'] > 0)]
            st.caption('You exclude ' + str(len(excluded)) + ' stocks of the SNB, '
                       '{:.1%} of its portfolio.'.format(excluded['SNB'].sum()))

    left, right = st.columns(2)
    with left:
        st.subheader('The cost of simplicity')
        with st.spinner('Computing...'):
            curve = pd.DataFrame({'Without exclusions': get_te_curve((), w_max, sector_tol, shrinkage, data_version)})
            if len(exclusions) > 0:
                curve['With your exclusions'] = get_te_curve(exclusions, w_max, sector_tol, shrinkage, data_version)
        fig = px.line(curve, markers = True, labels = {'index': 'Number of stocks', 'value': 'Expected tracking error',
                                                       'variable': ''},
                      color_discrete_sequence = [COLORS['Portfolio'], COLORS['SNB']])
        fig.update_layout(height = 330, yaxis_tickformat = '.1%', margin = dict(t = 10, b = 0),
                          legend = dict(orientation = 'h', y = 1.12))
        st.plotly_chart(fig)
        st.caption('Tracking error of the closest copy of the SNB for each number of stocks (last quarter). '
                   'Fewer stocks is simpler and cheaper, but further away from the SNB.')
    with right:
        st.subheader('Where does the tracking error come from ?')
        contributions = f.te_contributions(last['weights'].values, last['w_snb'].values, cov_matrices[-1])
        contributions = pd.Series(contributions, index = table['Sector']).groupby(level = 0).sum().sort_values()
        fig = px.bar(x = contributions.values, y = contributions.index, orientation = 'h',
                     labels = {'x': 'Contribution to the tracking error', 'y': ''})
        fig.update_traces(marker_color = COLORS['Portfolio'])
        fig.update_layout(height = 330, xaxis_tickformat = '.2%', margin = dict(t = 10, b = 0))
        st.plotly_chart(fig)
        st.caption('Risk contributions (Euler) of the tracking error by sector, they sum to the tracking error.')



# Tab 3 : backtest

with tab3:
    st.subheader('Value of 1 USD invested')
    cols = ['Portfolio', 'SNB', 'Market']
    wealth = (1 + daily[cols].fillna(0)).cumprod()
    fig = px.line(wealth, labels = {'index': '', 'date': '', 'value': 'USD', 'variable': ''},
                  color_discrete_map = COLORS)
    fig.update_layout(height = 420, margin = dict(t = 10, b = 0), legend = dict(orientation = 'h', y = 1.12))
    st.plotly_chart(fig)
    st.caption('We rebalance each quarter when the new 13F is public (45 days after the end of the quarter). '
               'The portfolio is net of trading costs. Market = the 500 largest US stocks weighted by market cap, '
               'dividends included (very close to the S&P 500 with dividends).')

    st.subheader('Performance')
    st.dataframe(perf.style.format('{:.2%}', na_rep = '-')
                 .format('{:.2f}', subset = ['Sharpe ratio', 'Information ratio'], na_rep = '-'))

    left, right = st.columns(2)
    with left:
        st.subheader('Tracking error over time')
        rolling_te = ((daily['Portfolio'] - daily['SNB']).rolling(252).std() * np.sqrt(252)).dropna()
        fig = px.line(x = rolling_te.index, y = rolling_te.values, labels = {'x': '', 'y': 'Tracking error (1 year)'})
        fig.update_traces(line_color = COLORS['Portfolio'])
        if te_budget > 0:
            fig.add_hline(y = te_budget, line_dash = 'dash', line_color = COLORS['SNB'], annotation_text = 'Your budget')
        fig.update_layout(height = 320, yaxis_tickformat = '.1%', margin = dict(t = 10, b = 0))
        st.plotly_chart(fig)
        st.caption('Shrinkage intensity of Ledoit-Wolf at the last rebalancing : δ = {:.0%}'.format(deltas[-1]))
    with right:
        st.subheader('Turnover at each rebalancing')
        fig = px.bar(results, x = 'rebalance_date', y = 'turnover', labels = {'rebalance_date': '', 'turnover': 'Turnover'})
        fig.update_traces(marker_color = COLORS['Portfolio'])
        if turnover_max is not None:
            fig.add_hline(y = turnover_max, line_dash = 'dash', line_color = COLORS['SNB'], annotation_text = 'Limit')
        fig.update_layout(height = 320, yaxis_tickformat = '.0%', margin = dict(t = 10, b = 0))
        st.plotly_chart(fig)
        st.caption('Average turnover : {:.0%} per quarter. '.format(results['turnover'].mean())
                   + 'When the bars touch the limit, the limit is binding : we would like to trade more. '
                   'The turnover can be above the limit when the limit is too small to respect the sector '
                   'constraints : we then trade the minimum needed.')