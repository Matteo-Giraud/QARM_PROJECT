import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import linprog


# Parameters of the strategy
EST_WINDOW = 504        # 2 years of daily returns to estimate the covariance matrix
MOM_WINDOW = 252        # momentum = return over the last 12 months...
MOM_SKIP = 21           # ...without the last month
SECTOR_TOL = 0.02       # by default, sector weights have to stay within +/- 2% of the SNB
TAU = 0.05              # same tau as in the Black-Litterman lab
SR_SNB = 0.25           # Sharpe ratio used for the implied returns (as in the lab)
VIEW_STRENGTH = 0.15    # a stock with 1 std of momentum above average gets +15% of its vol

# GICS codes of the sectors the client can exclude
EXCLUSIONS = {'Fossil fuels': ('gind', ['101020']),
              'Weapons & defense': ('gind', ['201010']),
              'Tobacco': ('gsubind', ['30203010']),
              'Alcohol': ('gsubind', ['30201010', '30201020']),
              'Gambling': ('gsubind', ['25301010'])}

SECTOR_NAMES = {'10': 'Energy', '15': 'Materials', '20': 'Industrials', '25': 'Consumer Discretionary',
                '30': 'Consumer Staples', '35': 'Health Care', '40': 'Financials', '45': 'Information Technology',
                '50': 'Communication Services', '55': 'Utilities', '60': 'Real Estate', '99': 'Unclassified'}



# 1. Covariance matrix


def ledoit_wolf(X):
    # Shrinkage of the sample covariance matrix towards the constant correlation matrix (LN3)
    # Formulas of Ledoit & Wolf (2003) "Honey, I shrunk the sample covariance matrix"
    T, N = X.shape
    X = X - X.mean(axis = 0)

    S = X.T @ X / T                                  # sample covariance matrix
    var = np.diag(S)
    std = np.sqrt(var)
    correl_matrix = S / np.outer(std, std)
    rho_bar = (np.sum(correl_matrix) - N) / (N * (N - 1))   # average correlation

    F = rho_bar * np.outer(std, std)                 # target with constant correlation
    np.fill_diagonal(F, var)

    # shrinkage constant (appendix of the paper)
    pi_mat = (X**2).T @ (X**2) / T - S**2
    pi_hat = np.sum(pi_mat)
    theta = (X**3).T @ X / T - var.reshape(-1, 1) * S
    np.fill_diagonal(theta, 0)
    rho_hat = np.sum(np.diag(pi_mat)) + rho_bar * np.sum(np.outer(1 / std, std) * theta)
    gamma_hat = np.sum((F - S)**2)
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = max(0, min(1, kappa / T))

    return delta * F + (1 - delta) * S, delta


def covariance_matrices(returns, quarters, shrinkage = True):
    # One annualized covariance matrix per rebalancing date, on the 2 years before it
    cov_matrices = []
    deltas = []
    for q in quarters:
        window = returns.loc[:q['rebalance_date'], q['permnos']].iloc[-EST_WINDOW - 1:-1]
        window = window.fillna(window.mean())       # a missing day is replaced by the average return of the stock
        if shrinkage:
            cov_matrix, delta = ledoit_wolf(window.values)
        else:
            cov_matrix, delta = np.cov(window.values, rowvar = False), 0
        cov_matrices.append(cov_matrix * 252)
        deltas.append(delta)
    return cov_matrices, deltas



# 2. Choice of the stocks


def is_excluded(sectors, exclusions):
    # True for the stocks that the client refuses (exclusions = list of keys of EXCLUSIONS)
    excluded = np.zeros(len(sectors), dtype = bool)
    for e in exclusions:
        level, codes = EXCLUSIONS[e]
        excluded = excluded | sectors[level].isin(codes).values
    return excluded


def select_stocks(w_snb, sectors, n):
    # Each sector gets a number of stocks proportional to its weight in the SNB portfolio,
    # then we take the largest SNB positions inside each sector
    sector_weights = pd.Series(w_snb).groupby(sectors).sum()
    sector_weights = sector_weights[sector_weights > 0] / sector_weights.sum()
    nb_stocks = np.floor(sector_weights * n).astype(int)

    # the stocks that are left go to the sectors with the largest decimal part
    rest = (sector_weights * n - nb_stocks).sort_values(ascending = False)
    for s in rest.index[:n - nb_stocks.sum()]:
        nb_stocks[s] += 1

    selected = []
    for s in nb_stocks.index:
        in_sector = np.where(sectors == s)[0]
        largest = in_sector[np.argsort(-w_snb[in_sector])]
        selected.extend(largest[:nb_stocks[s]])
    return np.sort(selected)



# 3. Optimization


def tracking_error(x, b, cov_matrix):
    # LN2 : volatility of the difference between the portfolio and the benchmark
    return np.sqrt((x - b) @ cov_matrix @ (x - b))


def QP_tracking_error(x, cov_matrix, b, idx):
    # We only hold the stocks in idx, the benchmark b (SNB) holds all of them.
    # x 10000 : the tracking error is very small and the optimizer stops too early otherwise
    w = np.zeros(len(b))
    w[idx] = x
    return 10000 * 0.5 * (w - b) @ cov_matrix @ (w - b)


def gradient_tracking_error(x, cov_matrix, b, idx):
    # Derivative of the function above = 10000 * Sigma (w - b), for the stocks we hold.
    # We give it to scipy, otherwise it computes it numerically and the backtest is much slower
    w = np.zeros(len(b))
    w[idx] = x
    return 10000 * (cov_matrix @ (w - b))[idx]


def black_litterman(cov_matrix, w_snb, idx, momentum, rf):
    # Same steps as in the Black-Litterman lab (LN4), with the SNB as initial portfolio x0
    implied_mu = rf + SR_SNB * (cov_matrix @ w_snb) / np.sqrt(w_snb @ cov_matrix @ w_snb)

    # views : one absolute view per stock we hold, based on its momentum
    vol = np.sqrt(np.diag(cov_matrix))
    z = (momentum[idx] - momentum[idx].mean()) / momentum[idx].std()
    z = np.clip(z, -3, 3)
    P = np.eye(len(w_snb))[idx]
    Q = implied_mu[idx] + VIEW_STRENGTH * z * vol[idx]

    gamma_matrix = TAU * cov_matrix
    omega = np.diag(np.diag(P @ gamma_matrix @ P.T))     # uncertainty of the views = uncertainty of the prior
    mu_bar = implied_mu + (gamma_matrix @ P.T) @ np.linalg.inv(P @ gamma_matrix @ P.T + omega) @ (Q - P @ implied_mu)
    return mu_bar


def optimize_portfolio(cov_matrix, w_snb, idx, sectors, w_max, sector_targets, mu = None, te_budget = 0,
                       x_prev = None, turnover_max = None, sector_tol = SECTOR_TOL):
    # Step 1 : weights of the stocks in idx with the smallest tracking error vs the SNB
    # Step 2 (only if mu is given) : maximize the expected return with a tracking error <= te_budget (LN2)
    # If x_prev is given we add the turnover limit like in LN3, with the variables X = (x, dx_minus, dx_plus)
    n = len(idx)
    w_max = max(w_max, 1 / n)          # otherwise we cannot be fully invested
    note = ''

    if x_prev is None:
        k = 0                          # only the n weights
    else:
        k = 2 * n                      # + the n dx_minus and the n dx_plus
    zeros = np.zeros(k)

    # 1. Fully invested weight constraint
    constraints = [LinearConstraint(np.concatenate([np.ones(n), zeros]), lb = 1, ub = 1)]

    # 2. Long only with a maximum weight per stock. Same as LinearConstraint(np.eye(n), lb = 0, ub = w_max)
    # but with "bounds" scipy is a lot faster
    bounds = [(0, w_max)] * n + [(0, 1)] * k

    # 3. Sector weights close to the SNB : one line of C per sector (Cx <= D in LN3)
    sector_list = np.unique(sectors[idx])
    C = np.zeros((len(sector_list), n + k))
    for i in range(len(sector_list)):
        C[i, :n] = (sectors[idx] == sector_list[i])
    targets = sector_targets[sector_list].values
    sector_constraint = LinearConstraint(C, lb = targets - sector_tol, ub = targets + sector_tol)

    # starting point : SNB weights of the stocks we hold, rescaled
    x0 = w_snb[idx] / np.sum(w_snb[idx])

    # 4. Turnover (LN3) : x + dx_minus - dx_plus = x_prev and sum(dx_minus + dx_plus) <= turnover_max
    if x_prev is not None:
        A = np.hstack([np.eye(n), np.eye(n), -np.eye(n)])
        dx = np.concatenate([np.zeros(n), np.ones(n), np.ones(n)])

        # Before, we check with a linear program the minimum turnover we need to respect the other
        # constraints. If the limit of the client is below, there is no solution (and scipy takes
        # ages to tell us), so we trade this minimum instead.
        lp = linprog(dx, A_ub = np.vstack([C, -C]),
                     b_ub = np.concatenate([targets + sector_tol, -(targets - sector_tol)]),
                     A_eq = np.vstack([np.concatenate([np.ones(n), zeros]), A]),
                     b_eq = np.concatenate([[1], x_prev]), bounds = bounds)
        if not lp.success:
            return None
        if lp.fun > turnover_max:
            turnover_max = lp.fun + 0.005
            note = 'turnover limit too small, we trade the minimum'

        constraints = constraints + [LinearConstraint(A, lb = x_prev, ub = x_prev),
                                     LinearConstraint(dx, ub = turnover_max)]
        x0 = np.concatenate([x0, np.maximum(x_prev - x0, 0), np.maximum(x0 - x_prev, 0)])

    def objective(X):
        return QP_tracking_error(X[:n], cov_matrix, w_snb, idx)

    def gradient(X):
        return np.concatenate([gradient_tracking_error(X[:n], cov_matrix, w_snb, idx), zeros])

    all_constraints = constraints + [sector_constraint]
    res = minimize(objective, x0, jac = gradient, bounds = bounds, options = {'disp': False, 'maxiter': 1000},
                   constraints = all_constraints)

    if not res.success:
        if x_prev is not None:
            return None
        # without turnover it only happens if the max weight is too small to reach the weight of a sector
        all_constraints = constraints
        res = minimize(objective, x0, jac = gradient, bounds = bounds, options = {'disp': False, 'maxiter': 1000},
                       constraints = all_constraints)
        note = 'sector constraints removed'
        if not res.success:
            note = 'the optimizer did not fully converge, weights are approximate'

    X = res.x
    te_copy = np.sqrt(2 * objective(X) / 10000)
    step = 1

    # Step 2 : we add our views if the budget leaves some room
    if mu is not None and te_budget > te_copy:
        mu_X = np.concatenate([mu[idx], zeros])

        def minus_expected_return(X):
            return -100 * X @ mu_X           # minus because we minimize, x 100 for the same reason as x 10000

        def gradient_return(X):
            return -100 * mu_X

        # 2 * objective = 10000 * TE^2, it has to stay below 10000 * te_budget^2
        def te_squared(X):
            return 2 * objective(X)

        def gradient_te_squared(X):
            return 2 * gradient(X)

        te_constraint = NonlinearConstraint(te_squared, lb = 0, ub = 10000 * te_budget**2, jac = gradient_te_squared)
        res2 = minimize(minus_expected_return, X, jac = gradient_return, bounds = bounds,
                        options = {'disp': False, 'maxiter': 1000}, constraints = all_constraints + [te_constraint])
        # Sometimes SLSQP stops with "Positive directional derivative for linesearch" when it is already
        # at the optimum and cannot do better. We checked : the weights are fine, so we keep them if the
        # tracking error respects the budget
        if res2.success or np.sqrt(te_squared(res2.x) / 10000) <= te_budget + 0.0001:
            X = res2.x
            step = 2

    x = np.clip(X[:n], 0, None)
    x = x / np.sum(x)
    return x, te_copy, step, note



# 4. Backtest


def buy_and_hold(period_returns, weights):
    # Daily returns of a portfolio that we do not rebalance during the quarter,
    # and the weights at the end of the quarter (they drift with the prices)
    R = period_returns[weights.index].fillna(0)
    growth = (1 + R).cumprod()
    value = growth @ weights
    daily_return = value / value.shift(1).fillna(1) - 1
    end_weights = weights * growth.iloc[-1] / value.iloc[-1]
    return daily_return, end_weights


def build_portfolio(k, data, cov_matrices, n_stocks, te_budget, exclusions, w_max,
                    turnover_max = None, previous_weights = None, sector_tol = SECTOR_TOL):
    # Portfolio of quarter k
    q = data['quarters'][k]
    cov_matrix = cov_matrices[k]
    permnos = q['permnos']
    w_snb = q['w_snb']
    sectors = q['sectors']['gsector'].values

    # stocks the client refuses get a weight of zero
    excluded = is_excluded(q['sectors'], exclusions)
    allowed = np.where(~excluded)[0]
    idx = allowed[select_stocks(w_snb[allowed], sectors[allowed], n_stocks)]

    # sector targets = SNB sector weights without the excluded stocks
    w_allowed = np.where(excluded, 0, w_snb)
    sector_targets = pd.Series(w_allowed).groupby(sectors).sum() / np.sum(w_allowed)

    # our views with Black-Litterman, only if the client accepts some tracking error
    mu = None
    if te_budget > 0:
        rets = data['returns'].loc[:q['rebalance_date'], permnos]
        momentum = (1 + rets.iloc[-MOM_WINDOW - 1:-MOM_SKIP - 1].fillna(0)).prod() - 1
        rf = data['market']['rf'].loc[:q['rebalance_date']].iloc[-1] * 252
        mu = black_litterman(cov_matrix, w_snb, idx, momentum.values, rf)

    # first we optimize without the turnover limit (n variables instead of 3n, much faster)
    x, te_copy, step, note = optimize_portfolio(cov_matrix, w_snb, idx, sectors, w_max, sector_targets, mu, te_budget,
                                                sector_tol = sector_tol)

    # if we trade too much, we optimize again with the turnover limit
    if previous_weights is not None and turnover_max is not None:
        x_prev = previous_weights.reindex(permnos[idx]).fillna(0).values
        # the stocks we drop are sold anyway, it already uses a part of the turnover
        sold = previous_weights.drop(permnos[idx], errors = 'ignore').sum()
        turnover_left = turnover_max - sold

        if np.sum(np.abs(x - x_prev)) > turnover_left:
            result = optimize_portfolio(cov_matrix, w_snb, idx, sectors, w_max, sector_targets, mu, te_budget,
                                        x_prev, turnover_left, sector_tol)
            if result is None:
                note = 'turnover limit removed'
            else:
                # we keep te_copy of the copy without turnover limit, as in the cost of simplicity curve
                x, _, step, note = result

    weights = pd.Series(0., index = permnos)
    weights.iloc[idx] = x

    return {'weights': weights, 'w_snb': pd.Series(w_snb, index = permnos),
            'w_mkt': pd.Series(q['w_mkt'], index = permnos), 'excluded': excluded,
            'te_copy': te_copy, 'te': tracking_error(weights.values, w_snb, cov_matrix),
            'step': step, 'note': note}


def backtest(data, cov_matrices, n_stocks, te_budget, exclusions, w_max, turnover_max, cost_bp,
             sector_tol = SECTOR_TOL):
    returns = data['returns']
    quarters = data['quarters']

    portfolio, snb, market, results = [], [], [], []
    previous_weights = None

    for k in range(len(quarters)):
        q = quarters[k]
        p = build_portfolio(k, data, cov_matrices, n_stocks, te_budget, exclusions, w_max,
                            turnover_max, previous_weights, sector_tol)

        # turnover = sum of the absolute changes of weights (LN3)
        if previous_weights is None:
            turnover = np.nan
        else:
            all_stocks = p['weights'].index.union(previous_weights.index)
            turnover = np.sum(np.abs(p['weights'].reindex(all_stocks, fill_value = 0)
                                     - previous_weights.reindex(all_stocks, fill_value = 0)))

        # We trade at the close of the rebalancing day, so returns start the day after
        start = q['rebalance_date']
        if k + 1 < len(quarters):
            end = quarters[k + 1]['rebalance_date']
            period = returns.loc[start:end].iloc[1:]
        else:
            period = returns.loc[start:].iloc[1:] 

        held = p['weights'][p['weights'] > 0]
        r_portfolio, previous_weights = buy_and_hold(period, held)
        r_snb, _ = buy_and_hold(period, p['w_snb'][p['w_snb'] > 0])
        r_market, _ = buy_and_hold(period, p['w_mkt'])

        # Trading costs are taken from the first day's return
        cost = 0 if np.isnan(turnover) else turnover * cost_bp / 10000
        r_portfolio.iloc[0] = r_portfolio.iloc[0] - cost

        portfolio.append(r_portfolio)
        snb.append(r_snb)
        market.append(r_market)
        results.append({'rebalance_date': start, 'report_date': q['report_date'], 'nb_stocks': len(held),
                        'te_copy': p['te_copy'], 'te': p['te'], 'step': p['step'], 'note': p['note'],
                        'turnover': turnover, 'cost': cost,
                        'te_snb_market': tracking_error(p['w_snb'].values, p['w_mkt'].values, cov_matrices[k])})

    daily = pd.DataFrame({'Portfolio': pd.concat(portfolio), 'SNB': pd.concat(snb), 'Market': pd.concat(market)})
    daily['rf'] = data['market']['rf'].reindex(daily.index).fillna(0)

    return daily, pd.DataFrame(results), p


def performance(daily):
    # Usual statistics, annualized with 252 days
    table = {}
    for col in ['Portfolio', 'SNB', 'Market']:
        r = daily[col].dropna()
        wealth = (1 + r).cumprod()
        annual_return = wealth.iloc[-1] ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = (r - daily['rf']).mean() * 252 / vol
        max_drawdown = (wealth / wealth.cummax() - 1).min()
        # tracking error and information ratio against the SNB (LN2)
        active = r - daily['SNB']
        te = active.std() * np.sqrt(252)
        ir = active.mean() * 252 / te if te > 0 else np.nan
        table[col] = [annual_return, vol, sharpe, max_drawdown, te, ir]
    table = pd.DataFrame(table, index = ['Annual return', 'Volatility', 'Sharpe ratio', 'Max drawdown',
                                         'Tracking error vs SNB', 'Information ratio']).T
    table.loc['SNB', ['Tracking error vs SNB', 'Information ratio']] = np.nan
    return table


def te_contributions(weights, w_snb, cov_matrix):
    # Risk contributions of the tracking error (same formula as LN5, with x - b instead of x)
    a = weights - w_snb
    return a * (cov_matrix @ a) / np.sqrt(a @ cov_matrix @ a)


def te_curve(data, cov_matrices, k, n_list, exclusions, w_max, sector_tol = SECTOR_TOL):
    # "Cost of simplicity" : tracking error of the copy for different numbers of stocks (quarter k)
    te = []
    for n in n_list:
        p = build_portfolio(k, data, cov_matrices, n, 0, exclusions, w_max, sector_tol = sector_tol)
        te.append(p['te'])
    return pd.Series(te, index = n_list)