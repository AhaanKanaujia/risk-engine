import datetime
import numpy as np
import pandas as pd

from risk import cache
from portfolio import portfolio
from portfolio import position

def calibrate_portfolio_gbm(pf: portfolio.Portfolio, date: datetime.date, window_size: int, lambda_val: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calibrate the stocks in portfolio to a geometric brownian motion model using historical price data up to the given date.
    Only considers last window_size days of price data for calibration, ignoring any data points before that.
    Weighted calibration gives more weight to recent price data, while unweighted calibration treats all price data equally.
    Also computes the correlation matrix between the stocks in the portfolio using the same historical price data.
    Returns the yearly drift and volatility parameters for each stock in the portfolio along with the correlation matrix between the stocks.

    Args:
        pf: The portfolio to calibrate.
        date: The date up to which to use historical price data for calibration.        
        window_size: The number of days of historical price data to use for calibration.
        lambda_val: The decay factor for weighted calibration (0 < lambda_val <= 1).
    
    Returns:
        A pandas DataFrame containing the calibrated parameters (drift and volatility) for each stock in the portfolio.
    """

    # get unique tickers from the positions
    tickers = pf.get_position_tickers_on_date(date)

    # load historical price data for the tickers
    price_data = {}
    for ticker in tickers:
        df = cache.get_price_history(ticker)
        price_data[ticker] = df.loc[df.index <= date]

    # calibrate the price date to a geometric brownian motion model for each ticker
    calibration_results = []
    # save log returns and weighted log returns of each stock to compute correlation matrix
    log_returns_df = pd.DataFrame()
    for ticker in tickers:
        # index by Date so returns align across tickers by date rather than by original integer index
        tckr_data = price_data[ticker].tail(window_size)
        log_returns = np.log(tckr_data["Close"] / tckr_data["Close"].shift(1)).dropna()

        n = len(log_returns)
        powers = np.arange(n - 1, -1, -1)
        weights = lambda_val ** powers
        weights /= weights.sum()

        log_returns_df[ticker] = log_returns

        mean_log_return = np.dot(weights, log_returns)
        volatility_log_return = np.dot(weights, (log_returns - mean_log_return) ** 2)

        num_days_per_year = 252 # constant number of trading days in a year
        mu = (mean_log_return + 0.5 * volatility_log_return) * num_days_per_year
        sigma = np.sqrt(volatility_log_return) * np.sqrt(num_days_per_year)

        calibration_results.append({
            "ticker": ticker,
            "drift": mu,
            "volatility": sigma
        })

    # compute correlation matrix between the stocks in the portfolio using the same historical price data
    # unweighted corr matrix using log returns, not used in the simulation
    corr_matrix = log_returns_df.corr()

    # compute weighted correlation matrix using same weights
    n = len(log_returns_df)
    powers = np.arange(n - 1, -1, -1)
    weights = lambda_val ** powers
    weights /= weights.sum()

    # mean eequivalent to np.dot(weights, log_returns_df)
    mean_vec = np.sum(log_returns_df * weights[:, np.newaxis], axis=0)
    centered_log_returns = log_returns_df - mean_vec
    weighted_cov_matrix = (centered_log_returns.T * weights) @ centered_log_returns

    std_dev = np.sqrt(np.diag(weighted_cov_matrix))
    weighted_corr_matrix = weighted_cov_matrix / np.outer(std_dev, std_dev)

    return pd.DataFrame(calibration_results), weighted_corr_matrix
