import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio import portfolio
from models import monte_carlo

def plot_simulated_portfolio_value(paths: np.ndarray):
    """
    Plot the simulated portfolio value paths.

    Args:
        paths: A numpy array of shape (num_simulations, up_to_days) containing the simulated portfolio value paths.
    """

    num_simulations, num_days = paths.shape
    days = np.arange(num_days)

    plt.figure(figsize=(10, 6))
    sim_color = "orange"
    stat_color = "black"

    # plot all simulation lines in blue, faint for visibility
    plt.plot(days, paths.T, color=sim_color, alpha=0.25)

    # compute summary statistics across simulations
    mean_path = paths.mean(axis=0)
    lower = np.percentile(paths, 2.5, axis=0)
    upper = np.percentile(paths, 97.5, axis=0)

    # plot mean path in black
    plt.plot(days, mean_path, color=stat_color, linewidth=2, alpha=0.25, label="Mean")

    # 95% confidence band in black (semi-transparent)
    plt.fill_between(days, lower, upper, color=stat_color, alpha=0.1, label="95% CI")

    plt.title(f"Simulated Portfolio Value Paths — {num_simulations} sims")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")

    # extra info in the legend / annotation
    start_mean = paths[:, 0].mean()
    end_mean = mean_path[-1]
    annotation = f"Start mean: {start_mean:.2f}\nEnd mean: {end_mean:.2f}\nSimulations: {num_simulations}"
    plt.gca().text(0.98, 0.02, annotation, transform=plt.gca().transAxes,
                   ha="right", va="bottom", fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def compute_portfolio_monte_carlo_var_es(pf: portfolio.Portfolio, date: datetime.date, window_size: int, up_to_days: int, alpha: float, lambda_val: float, num_simulations: int) -> tuple[np.ndarray, float, float]:
    """
    Computes the Monte Carlo VaR and ES of portfolio on the given date using the last window_size days of historical price data to calibrate a geometric brownian motion model for each stock in the portfolio and simulating future stock price paths.
    VaR is the maximum loss that the portfolio could have experienced with a confidence level of alpha over the next up_to_days days.
    ES is the expected loss that the portfolio could have experienced with a confidence level of alpha over the next up_to_days days.

    Args:
        pf: The portfolio to compute Monte Carlo VaR for.
        date: The date on which to compute the Monte Carlo VaR.
        window_size: The number of days of historical price data to use for calibrating the geometric brownian motion model.
        up_to_days: The number of days in the future to compute the Monte Carlo VaR for.
        lambda_val: The lambda parameter for the calibration function.
        alpha: The confidence level for the Monte Carlo VaR calculation (e.g. 0.95 for 95% confidence level).
        num_simulations: The number of simulated paths to generate for each stock in the portfolio.
    
    Returns:
        The Monte Carlo VaR and ES of the portfolio on the given date.
    """

    # compute future portfolio value paths using the simulated stock price paths and current portfolio quantities
    portfolio_value_paths = monte_carlo.simulate_portfolio_value(pf, date, window_size, up_to_days, lambda_val=lambda_val, num_simulations=num_simulations)

    # get only the portfolio value on last day, rest of the days are not relevant for computing the ES for up_to_days days
    first_day_portfolio_value_paths = portfolio_value_paths[:, 0]
    last_day_portfolio_value_paths = portfolio_value_paths[:, -1]

    # get pnl for last day of the simulated paths as difference between simulated portfolio value and portfolio value on first day of the simulated paths
    pnl_paths = last_day_portfolio_value_paths - first_day_portfolio_value_paths
    
    # compute the VaR at the given confidence level from the distribution of simulated PnL values
    var = -np.percentile(pnl_paths.flatten(), (1 - alpha) * 100) # negative sign to convert from loss to positive VaR value

    # compute the ES at the given confidence level from the distribution of simulated PnL values
    es = -np.mean([pnl for pnl in pnl_paths.flatten() if pnl <= -var]) # negative sign to convert from loss to positive ES value

    return pnl_paths, var, es
