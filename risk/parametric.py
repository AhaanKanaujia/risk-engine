import os
import warnings
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm

from portfolio import portfolio, position
from models import calibration
from pricing import black_scholes
from risk import utils

def get_option_position_deltas(
    option_positions: list[position.Position],
    date: datetime.date,
) -> dict[str, float]:
    """
    Compute the delta of each option position in the portfolio on the given date using the Black-Scholes model.

    Args:
        option_positions: A list of option positions in the portfolio.
        date: The date on which to compute the option deltas.

    Returns:
        A dictionary mapping each option position ticker to its corresponding delta value.
    """

    option_deltas = {}
    for pos in option_positions:
        underlying_ticker = pos.underlying # get underlying ticker of the option from the position data
        expiry = pos.expiry # get expiry date of the option from the position data
        option_type = pos.option_type # get option type (call or put) from the position data
        strike = pos.strike # get strike price of the option from the position data
        implied_vol = black_scholes.option_implied_vol(date, underlying_ticker, expiry, option_type, strike)
        maturity_in_days = (expiry - date).days # get number of days till maturity from the given date
        time_to_maturity_in_years = maturity_in_days / 252 # convert time to maturity to years for the Black-Scholes formula

        # compute delta of the option using the black scholes model
        delta = black_scholes.option_delta(
            S=pos.get_underlying_price_on_date(date),
            K=strike,
            T=time_to_maturity_in_years,
            r=utils.get_risk_free_rate(date),
            sigma=implied_vol,
            option_type=option_type
        )
        option_deltas[pos.ticker] = delta
    
    return option_deltas

def get_option_position_gammas(
    option_positions: list[position.Position],
    date: datetime.date,
) -> dict[str, float]:
    """
    Compute the gamma of each option position in the portfolio on the given date using the Black-Scholes model.

    Args:
        option_positions: A list of option positions in the portfolio.
        date: The date on which to compute the option gammas.

    Returns:
        A dictionary mapping each option position ticker to its corresponding gamma value.
    """

    option_gammas = {}
    for pos in option_positions:
        underlying_ticker = pos.underlying # get underlying ticker of the option from the position data
        expiry = pos.expiry # get expiry date of the option from the position data
        option_type = pos.option_type # get option type (call or put) from the position data
        strike = pos.strike # get strike price of the option from the position data
        implied_vol = black_scholes.option_implied_vol(date, underlying_ticker, expiry, option_type, strike)
        maturity_in_days = (expiry - date).days # get number of days till maturity from the given date
        time_to_maturity_in_years = maturity_in_days / 252 # convert time to maturity to years for the Black-Scholes formula

        # compute gamma of the option using the black scholes model
        gamma = black_scholes.option_gamma(
            S=pos.get_underlying_price_on_date(date),
            K=strike,
            T=time_to_maturity_in_years,
            r=utils.get_risk_free_rate(date),
            sigma=implied_vol
        )
        option_gammas[pos.ticker] = gamma
    
    return option_gammas

def compute_portfolio_value_distribution(
    pf: portfolio.Portfolio,
    date: datetime.date,
    window_size: int,
    up_to_days: int,
    lambda_val: float,
    manual_return_moments: dict[str, dict[str, float]] | None = None,
) -> tuple[float, float, float]:
    """
    Compute the distribution of portfolio values for the portfolio on the given date using the last window_size days of historical price data.
    Computes price changes for each stock and entire portfolio over up_to_days days.

    Args:
        pf: The portfolio to compute the value distribution for.
        date: The date on which to compute the value distribution.
        window_size: The number of days of historical price data to use for computing the value distribution.
        up_to_days: The number of days in the future to compute the value distribution for.
        lambda_val: The lambda parameter for the calibration function.
        manual_return_moments: Optional dictionary of user-specified return means
            and variances for each stock ticker.

    Returns:
        A tuple containing the mean and standard deviation of the portfolio value distribution and the current portfolio value on the given date.
    """

    # calibrate the price date to a geometric brownian motion model for each ticker
    calib_results, corr_results = calibration.calibrate_portfolio_gbm(
        pf,
        date,
        window_size,
        lambda_val=lambda_val,
        manual_return_moments=manual_return_moments,
    )
    mu = np.asarray(calib_results.set_index("ticker")["drift"].to_numpy())
    sigma = np.asarray(calib_results.set_index("ticker")["volatility"].to_numpy())
    corr_matrix = np.asarray(corr_results.to_numpy())
    cov_matrix = np.outer(sigma, sigma) * corr_matrix

    # compute delta and gamma exposure of options
    option_positions = pf.get_portfolio_option_positions_on_date(date)
    option_deltas = get_option_position_deltas(option_positions, date)
    option_gammas = get_option_position_gammas(option_positions, date)

    # get portfolio weights based on current prices and quantities of the stocks in the portfolio
    # adjust for delta exposure from options (assume delta stocks equivalent to the option position)
    prices = pf.get_position_prices_on_date(date)
    stock_quantities = pf.get_portfolio_stock_quantities_on_date(date)
    option_quantities = pf.get_portfolio_option_quantities_on_date(date)

    tickers = pf.get_position_tickers_on_date(date)
    stock_positions = pf.get_portfolio_stock_positions_on_date(date)
    option_positions = pf.get_portfolio_option_positions_on_date(date)

    # compute portfolio dollar pnl distribution
    D = np.zeros(len(tickers)) # dollar change in position for 1 unit change in price of underlying
    G = np.zeros(len(tickers)) # dollar change in delta of position for 1 unit change in price of underlying

    for idx, ticker in enumerate(tickers):
        # stocks only contribute to first order price changes
        for pos in stock_positions:
            if pos.ticker == ticker:
                D[idx] += prices[ticker] * stock_quantities[ticker]

        # options contribute to first and second order price changes
        for pos in option_positions:
            if pos.underlying == ticker:
                D[idx] += option_deltas[pos.ticker] * prices[ticker] * option_quantities[pos.ticker]
                G[idx] += option_gammas[pos.ticker] * (prices[ticker] ** 2) * option_quantities[pos.ticker]

    # compute portfolio mean and variance using the calibrated parameters and correlation matrix
    # take into account delta and gamma exposure of options into portfolio mean and variance
    num_days_per_year = 252
    dt = up_to_days / num_days_per_year
    up_to_days_mu = (mu - 0.5 * sigma ** 2) * dt
    up_to_days_volatility = sigma ** 2 * dt

    mean_delta = np.dot(D, up_to_days_mu) # contribution of first order (delta) price change to portfolio mean
    mean_gamma = 0.5 * np.sum(G * up_to_days_volatility) # contribution of second order (gamma) price change to portfolio mean
    portfolio_mean = mean_delta + mean_gamma

    up_to_days_cov_matrix = cov_matrix * dt # scale covariance matrix to up_to_days days using the square root of time rule
    variance_delta = D @ up_to_days_cov_matrix @ D.T # contribution of first order (delta)
    variance_gamma = 0.5 * (G @ (up_to_days_cov_matrix ** 2) @ G.T) # contribution of second order (gamma)
    portfolio_variance = variance_delta + variance_gamma
    portfolio_sigma = np.sqrt(portfolio_variance)
    
    return portfolio_mean, portfolio_sigma

def compute_portfolio_parametric_var_es(
    pf: portfolio.Portfolio,
    date: datetime.date,
    window_size: int,
    up_to_days: int,
    alpha: float,
    lambda_val: float,
    manual_return_moments: dict[str, dict[str, float]] | None = None,
) -> tuple[np.float, np.float]:
    """
    Compute the parametric VaR and ES of portfolio on the given date using the last window_size days of historical price data to compute the mean and volatility of the portfolio value distribution.
    VaR is the maximum loss that the portfolio could have experienced with a confidence level of alpha over the next up_to_days days.
    ES is the expected loss that the portfolio could have experienced with a confidence level of alpha over the next up_to_days days.

    Args:
        pf: The portfolio to compute parametric VaR for.
        date: The date on which to compute the parametric VaR.
        window_size: The number of days of historical price data to use for computing the mean and volatility of the portfolio value distribution.
        up_to_days: The number of days in the future to compute the parametric VaR for.
        lambda_val: The lambda parameter for the calibration function.
        alpha: The confidence level for the parametric VaR calculation (e.g. 0.95 for 95% confidence level).
        manual_return_moments: Optional dictionary of user-specified return means
            and variances for each stock ticker.
    
    Returns:
        The parametric VaR of the portfolio on the given date.
    """

    portfolio_mu, portfolio_sigma = compute_portfolio_value_distribution(
        pf,
        date,
        window_size,
        up_to_days,
        lambda_val,
        manual_return_moments=manual_return_moments,
    )

    z = norm.ppf(1 - alpha)
    var = -(portfolio_mu + z * portfolio_sigma)
    es = -(portfolio_mu - (norm.pdf(z) / (1 - alpha)) * portfolio_sigma)

    return var, es
