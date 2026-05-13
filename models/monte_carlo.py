import os
import datetime
from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from risk import utils
from portfolio import portfolio, position
from models import calibration, test

from pricing import black_scholes

def plot_simulated_paths(
    paths: dict[str, np.ndarray],
):
    """
    Plot the simulated stock price paths for each ticker.

    Args:
        paths: A dictionary mapping each ticker to a numpy array of shape (num_simulations, up_to_days) containing the simulated stock price paths.
    """

    for ticker, sim_paths in paths.items():
        # sim_paths shape: (num_simulations, up_to_days)
        num_simulations, num_days = sim_paths.shape
        days = np.arange(num_days)

        plt.figure(figsize=(10, 6))
        sim_color = "orange"
        stat_color = "black"

        # plot all simulation lines in orange, faint for visibility
        plt.plot(days, sim_paths.T, color=sim_color, alpha=0.25)

        # compute summary statistics across simulations
        mean_path = sim_paths.mean(axis=0)
        lower = np.percentile(sim_paths, 2.5, axis=0)
        upper = np.percentile(sim_paths, 97.5, axis=0)

        # plot mean path in black
        plt.plot(days, mean_path, color=stat_color, linewidth=2, alpha=0.25, label="Mean")

        # 95% confidence band in black (semi-transparent)
        plt.fill_between(days, lower, upper, color=stat_color, alpha=0.1, label="95% CI")

        plt.title(f"Simulated Stock Price Paths for {ticker} — {num_simulations} sims")
        plt.xlabel("Days")
        plt.ylabel("Price")

        # extra info in the legend / annotation
        start_mean = sim_paths[:, 0].mean()
        end_mean = mean_path[-1]
        annotation = f"Start mean: {start_mean:.2f}\nEnd mean: {end_mean:.2f}\nSimulations: {num_simulations}"
        plt.gca().text(0.98, 0.02, annotation, transform=plt.gca().transAxes,
                       ha="right", va="bottom", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
def simulate_stock_prices(
    pf: portfolio.Portfolio,
    date: datetime.date,
    window_size: int,
    up_to_days: int,
    lambda_val: float,
    num_simulations: int,
    manual_return_moments: dict[str, dict[str, float]] | None = None,
    perform_statistical_test: bool = False,
) -> dict[str, np.ndarray]:
    """
    Simulate the future stock price paths for stocks in portfolio using geometric brownian motion model.
    Use calibration function to get the drift and volatility parameters for each stock in the portfolio on given date.
    Simulate num_simulations paths for each stock over the next up_to_days days.

    Args:
        pf: The portfolio to simulate stock price paths for.
        date: The date on which to calibrate the model and start the simulations.
        window_size: The number of days of historical price data to use for calibration.
        up_to_days: The number of future days to simulate stock price paths for.
        lambda_val: The lambda parameter for the calibration function.
        num_simulations: The number of simulated paths to generate for each stock.
        manual_return_moments: Optional dictionary of user-specified return means
            and variances for each stock ticker.
        perform_statistical_test: Whether to perform a statistical test on the simulated paths.

    Returns:
        A dictionary mapping each ticker to a numpy array of shape (num_simulations, up_to_days+1) containing the simulated stock price paths.
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

    # ensure a stable ticker order and get prices in that order
    tickers = list(pf.get_position_tickers_on_date(date))
    prices_map = pf.get_position_prices_on_date(date)
    start_prices = np.array([prices_map[t] for t in tickers])

    # simulate stock price paths for each ticker assuming gbm with correlation between the stocks
    n, _ = corr_matrix.shape
    num_days_per_year = 252
    dt = 1 / num_days_per_year

    # generate independent normal random variables for each stock and each simulation
    Z = np.random.normal(size=(num_simulations, up_to_days, n)).astype(np.float64)

    # generate correlated random variables using the cholesky decomposition
    # ensure the correlation matrix is positive definite for cholesky decomposition
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # add a small value to the diagonal to make it positive definite
        corr_matrix += np.eye(n) * 1e-6
        L = np.linalg.cholesky(corr_matrix)
    corr_Z = Z @ L.T

    # get drift and noise terms for each stock and each simulation
    drift = (mu - 0.5 * sigma ** 2) * dt
    noise = sigma * np.sqrt(dt) * corr_Z
    returns = drift + noise

    # get cummulative returns for each stock and each simulation
    cum_log_returns = np.cumsum(returns, axis=1)

    # get simulated log paths for each stock and each simulation
    log_paths = np.concatenate([
        np.zeros((num_simulations, 1, n)),
        cum_log_returns
    ], axis=1)

    paths = start_prices * np.exp(log_paths)

    if perform_statistical_test:
        # peform sample test to check if the simulated log returns have the same distribution as the historical log returns used for calibration
        check = test.test_correlation_matrix(paths, corr_matrix)
        print("Number of samples:", check["N_samples"])
        print("Expected error scale:", check["expected_error_scale"])
        print("Max abs correlation error:", check["max_abs_error"])
        print("Max test statistic:", check["max_test_stat"])
        print("Min p-value:", check["min_p_value"])
        print("Reject null hypothesis:", check["reject_null"])

    return {ticker: paths[:, :, i] for i, ticker in enumerate(tickers)}

def simulate_option_prices(
    pf: portfolio.Portfolio,
    date: datetime.date,
    stock_price_paths: dict[str, np.ndarray],
    option_positions: list[position.Position],
    window_size: int,
    lambda_val: float,
) -> dict[str, np.ndarray]:
    """
    Simulate the future option price paths for options in portfolio using the simulated stock price paths for the underlying stocks.
    Use Black-Scholes formula to compute option prices from the simulated stock price paths.

    Args:
        pf: The portfolio object containing the options to simulate.
        date: The date on which to start the simulations (used to compute time to maturity for the options).
        stock_price_paths: A dictionary mapping each ticker to a numpy array of shape (num_simulations, up_to_days+1) containing the simulated stock price paths for the underlying stocks.
        option_positions: A list of option positions in the portfolio to simulate option price paths for.
        window_size: The number of days of historical price data to use for calibration.
        lambda_val: The lambda parameter for the calibration function.

    Returns:
        A dictionary mapping each option ticker to a numpy array of shape (num_simulations, up_to_days) containing the simulated option price paths.
    """

    option_price_paths = {}

    for pos in option_positions:
        underlying_ticker = pos.underlying # get underlying ticker of the option from the position data
        expiry = pos.expiry # get expiry date of the option from the position data
        option_type = pos.option_type # get option type (call or put) from the position data
        strike = pos.strike # get strike price of the option from the position data

        # simulate implied vol using calibration function for up to days, instead of just using fixed implied vol
        implied_vol = black_scholes.option_implied_vol(date, underlying_ticker, expiry, option_type, strike)

        maturity_in_days = (expiry - date).days # get number of days till maturity from the given date

        if underlying_ticker not in stock_price_paths:
            raise ValueError(f"Stock price paths for underlying ticker {underlying_ticker} not found. Cannot simulate option price paths for option {pos.ticker}.")
        
        # get the simulated stock price paths for the underlying ticker
        underlying_paths = stock_price_paths[underlying_ticker]

        # compute the time to maturity for each day in the simulation
        num_simulations, up_to_days = underlying_paths.shape
        days = np.arange(up_to_days)
        time_to_maturity_in_days = np.maximum(maturity_in_days - days, 0) # get time to maturity in days for each day in the simulation, capped at 0 after maturity
        time_to_maturity_in_years = time_to_maturity_in_days / 252 # convert time to maturity to years for the Black-Scholes formula

        # compute the option price paths using the Black-Scholes formula
        option_paths = black_scholes.option_price(
            S=underlying_paths,
            K=pos.strike,
            T=time_to_maturity_in_years,
            r=utils.get_risk_free_rate(date),
            sigma=implied_vol,
            option_type=pos.option_type
        )

        option_price_paths[pos.ticker] = option_paths

    return option_price_paths

def simulate_portfolio_value(
    pf: portfolio.Portfolio,
    date: datetime.date,
    window_size: int,
    up_to_days: int,
    lambda_val: float,
    num_simulations: int,
    manual_return_moments: dict[str, dict[str, float]] | None = None,
    perform_statistical_test: bool = False,
) -> np.ndarray:
    """
    Simulate the future portfolio value paths using the simulated stock price paths for the stocks in the portfolio.

    Args:
        pf: The portfolio to simulate value paths for.
        date: The date on which to calibrate the model and start the simulations.
        window_size: The number of days of historical price data to use for calibration.
        up_to_days: The number of future days to simulate portfolio value paths for.
        lambda_val: The lambda parameter for the calibration function.
        num_simulations: The number of simulated paths to generate for each stock.
        manual_return_moments: Optional dictionary of user-specified return means
            and variances for each stock ticker.
        perform_statistical_test: Whether to perform a statistical test on the simulated paths.
    Returns:
        A numpy array of shape (num_simulations, up_to_days) containing the simulated portfolio value paths.
    """

    # get stock positions pnl
    stock_price_paths = simulate_stock_prices(
        pf,
        date,
        window_size,
        up_to_days,
        lambda_val=lambda_val,
        num_simulations=num_simulations,
        manual_return_moments=manual_return_moments,
        perform_statistical_test=perform_statistical_test,
    )

    # get stock positions in the portfolio
    stock_positions = pf.get_portfolio_stock_positions_on_date(date)

    # get tickers of stock positions in the portfolio
    tickers = [pos.ticker for pos in stock_positions]

    if tickers:
        # get quantity of each stock position in the portfolio
        quantities = pf.get_portfolio_stock_quantities_on_date(date)

        # stack into shape (num_simulations, up_to_days, n)
        stock_prices = np.stack([stock_price_paths[t] for t in tickers], axis=2)
        stock_quantities = np.array([quantities[t] for t in tickers])

        # compute portfolio value for each simulation and each day as sum of quantity * stock price across all stocks in the portfolio
        stock_portfolio_value_paths = np.sum(stock_prices * stock_quantities, axis=2)
    else:
        # simulate zero stock portfolio value paths if there are no stock positions in the portfolio
        stock_portfolio_value_paths = np.zeros((num_simulations, up_to_days+1))

    # get option positions pnl
    option_positions = pf.get_portfolio_option_positions_on_date(date)

    # get quantity of each option position in the portfolio
    option_quantities = pf.get_portfolio_option_quantities_on_date(date)

    # compute option values for each simulation and each day using the simulated stock price paths
    option_price_paths = simulate_option_prices(pf, date, stock_price_paths, option_positions, window_size, lambda_val)

    # compute portfolio value for each simulation and each day as sum of quantity * option price across all options in the portfolio
    option_portfolio_value_paths = np.zeros_like(stock_portfolio_value_paths)
    for pos in option_positions:
        if pos.ticker not in option_price_paths:
            raise ValueError(f"Option price paths for option ticker {pos.ticker} not found. Cannot compute portfolio value paths.")
        option_paths = option_price_paths[pos.ticker]
        option_portfolio_value_paths += option_paths * option_quantities[pos.ticker]

    return stock_portfolio_value_paths + option_portfolio_value_paths
