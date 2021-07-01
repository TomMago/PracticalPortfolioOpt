'''
Basic methods for portfolio optimization
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize, Bounds, LinearConstraint

def returns(stocks):
    '''Calculates the returns in percent from stock tick data

    Args:
        stocks (pxn numpy array): time series of p number of stocks
                                  with length n

    Returns:
        stocks_retuns (pxn numpy array): time series of returns
    '''
    return stocks[:, 1:] / stocks[:, :-1]


def mean(stocks):
    '''Calculates the means for given stocks

    Args:
        stocks (pxn numpy array): time series of returns of p number of stocks
                                  with length n

    Returns:
        means (p numpy array): vector of means of all stocks
    '''
    means = np.mean(stocks, axis=1)
    return means


def cov_matrix(stocks):
    '''Calculates the covariance matrix for given stocks

    Args:
        stocks (pxn numpy array): time series of returns of p number of stocks
                                  with length n

    Returns:
        cov (pxp numpy array): cov matrix of stock returns
    '''
    cov = np.cov(stocks)
    return cov


def plug_in_allocation(stocks, risk_level):
    '''Calculates the optimal portfolio allocation with plug in estimator

    Args:
        stocks (pxn numpy array): Time series of returns of p number of stocks
                                  with length n
        risk_level (float): Given level of risk the allocation should have

    Returns:
        allocation (p numpy array): Percent of capital to keep in one asset
        returns (float): The return of the given allocation on historical
                         data
    '''
    num_stocks = stocks.shape[0]
    means = mean(stocks)
    cov = cov_matrix(stocks)
    cov_inv = np.linalg.inv(cov)

    normalization = np.sqrt(np.matmul(means.transpose(),
                                      np.matmul(cov_inv, means)))
    first_alloc = risk_level * cov_inv.dot(means) / normalization

    if sum(first_alloc) < 1:
        allocation = first_alloc
    else:
        first_term = cov_inv.dot(np.ones(num_stocks)) / np.sum(cov_inv)
        side_term = np.matmul(np.ones(num_stocks).transpose(),
                              np.matmul(cov_inv, means))
        b_term = np.sqrt((np.sum(cov_inv) * risk_level**2 - 1)
                         / (normalization**2 * np.sum(cov_inv) - side_term**2))
        last_term = side_term / np.sum(cov_inv) * np.matmul(cov_inv, np.ones(num_stocks))
        allocation = first_term + b_term * (np.matmul(cov_inv, means)
                                            - last_term)

    mean_returns = allocation.dot(means)

    return allocation, mean_returns

def noshort_allocation(stocks, risk_level):
    '''
    Calculates the optimal portfolio allocation where all the weight are between 0 and 1

    Args:
        stocks (pxn numpy array): Time series of returns of p number of stocks
                                  with length n
        risk_level (float): Given level of risk the allocation should have

    Returns:
        allocation (p numpy array): Percent of capital to keep in one asset
        returns (float): The return of the given allocation on historical
                         data
    '''
    num_stocks = stocks.shape[0]
    means = mean(stocks)
    cov = cov_matrix(stocks)

    inital_weight = np.ones((num_stocks, 1))/num_stocks

    bounds = Bounds(0, 1)

    opt_constraints = ({'type': 'eq',
                        'fun': lambda c: 1.0 - np.sum(c)},
                       {'type': 'ineq',
                        'fun': lambda c: risk_level**2 - c.T@cov@c})

    optimal_weights = minimize(lambda c: -c.T@means, inital_weight,
                               method='SLSQP',
                               bounds=bounds,
                               constraints=opt_constraints)

    mean_returns = optimal_weights['x'].dot(means)

    return optimal_weights['x'], mean_returns


def bootstraped_allocation(stocks, risk_level):
    '''Calculates the optimal portfolio allocation with bootstrapping procedure

    Args:
        stocks (pxn numpy array): Time series of returns of p number of stocks
                                  with length n
        risk_level (float): Given level of risk the allocation should have

    Returns:
        allocation (p numpy array): Percent of capital to keep in one asset
        returns (float): The return of the given allocation on historical
                         the data
    '''
    num_stocks = stocks.shape[0]
    num_samples = stocks.shape[1]
    means = mean(stocks)
    cov = cov_matrix(stocks)

    resample = np.random.multivariate_normal(means, cov, size=(num_samples))
    resample = resample.transpose()

    plug_in_c, plug_in_ret = plug_in_allocation(stocks, risk_level)
    resample_c, resample_ret = plug_in_allocation(resample, risk_level)

    gamma = 1/(1 - num_stocks/num_samples)

    allocation = plug_in_c + 1/np.sqrt(gamma) * (plug_in_c - resample_c)
    mean_return = plug_in_ret + 1/np.sqrt(gamma) * (plug_in_ret - resample_ret)

    return allocation, mean_return


def noshort_bootstraped_allocation(stocks, risk_level):
    '''Calculates the optimal portfolio allocation for positive weights with bootstrapping procedure

    Args:
        stocks (pxn numpy array): Time series of returns of p number of stocks
                                  with length n
        risk_level (float): Given level of risk the allocation should have

    Returns:
        allocation (p numpy array): Percent of capital to keep in one asset
        returns (float): The return of the given allocation on historical
                         the data
    '''
    num_stocks = stocks.shape[0]
    num_samples = stocks.shape[1]
    means = mean(stocks)
    cov = cov_matrix(stocks)

    resample = np.random.multivariate_normal(means, cov, size=(num_samples))
    resample = resample.transpose()

    plug_in_c, plug_in_ret = noshort_allocation(stocks, risk_level)
    resample_c, resample_ret = noshort_allocation(resample, risk_level)

    gamma = 1/(1 - num_stocks/num_samples)

    allocation = plug_in_c + 1/np.sqrt(gamma) * (plug_in_c - resample_c)
    mean_return = plug_in_ret + 1/np.sqrt(gamma) * (plug_in_ret - resample_ret)

    return allocation, mean_return


def efficency_curve(stocks, lower_risk=None, upper_risk=None, plot_points=500):
    '''
    Plots the efficency frontier

    Args:
        stocks (pxn numpy array): Time series of returns of p number of stocks
                                  with length n
    '''

    stock_stds = cov_matrix(stocks).diagonal()**0.5

    if not lower_risk:
        lower_risk = min(stock_stds)
    if not upper_risk:
        upper_risk = max(stock_stds)

    risk_points = np.linspace(lower_risk, upper_risk, plot_points)

    bootstrapped_returns = []
    plug_returns = []

    for i in risk_points:
        _, ret = bootstraped_allocation(stocks, i)
        _, ret_plug = plug_in_allocation(stocks, i)
        bootstrapped_returns.append(ret)
        plug_returns.append(ret_plug)

    def fit(x, a, b, c):
        return a*x**2 + b*x + c

    params, _ = curve_fit(fit, risk_points, bootstrapped_returns)

    plt.plot(risk_points, bootstrapped_returns)
    plt.plot(risk_points, plug_returns)
    plt.plot(risk_points, fit(risk_points, *params))


def noshort_efficency_curve(stocks, lower_risk=None, upper_risk=None, plot_points=200):
    '''
    Plots the efficency frontier for only positive weights

    For large risk values the curve will be constant since the optimal allocation with maximum risk
    ist reached (weight 1 for best performing stock)

    Args:
        stocks (pxn numpy array): Time series of returns of p number of stocks
                                  with length n
    '''

    stock_stds = cov_matrix(stocks).diagonal()**0.5

    if not lower_risk:
        lower_risk = min(stock_stds)
    if not upper_risk:
        upper_risk = max(stock_stds)/2

    risk_points = np.linspace(lower_risk, upper_risk, plot_points)

    bootstrapped_returns = []
    noshort_returns = []

    for i in risk_points:
        _, ret = noshort_allocation(stocks, i)
        _, ret_plug = noshort_bootstraped_allocation(stocks, i)
        bootstrapped_returns.append(ret_plug)
        noshort_returns.append(ret)

    plt.plot(risk_points, noshort_returns)
    plt.plot(risk_points, bootstrapped_returns)
