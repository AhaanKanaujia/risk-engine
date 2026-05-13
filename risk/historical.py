import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio import portfolio
from models import monte_carlo
from pricing import black_scholes
import risk.utils

def get_stock_portfolio_pnl_distribution(
    pf: portfolio.Portfolio,
    date: datetime.date,
    window_size: int,
    up_to_days: int,
) -> pd.Series:
    """
    Compute the distribution of PnL for the stock positions in the portfolio on the given date using the last window_size days of historical price data.
    Computes price changes for each stock and entire portfolio over up_to_days days.

    Args:
        pf: The portfolio to compute the PnL distribution for.
        date: The date on which to compute the PnL distribution.
        window_size: The number of days of historical price data to use for computing the PnL distribution.
        up_to_days: The number of days in the future to compute the PnL distribution for.

    Returns:
        A dictionary mapping dates to the corresponding PnL values.
    """

    # using price changes over up_to_days days, so load prices with up_to_days included in the window size
    # ensures that we have the necessary price data to compute price changes over up_to_days days, which is needed to compute the PnL distribution for up_to_days days
    tickers, stock_prices, stock_quantities = risk.utils.load_stock_prices_quantities(pf, date, window_size, up_to_days)

    if not stock_prices.empty:
        # compute price changes over up_to_days days as (current - previous) for each stock, then compute stock pnl = qty * change, summed over tickers
        stock_price_changes = stock_prices.diff(up_to_days).iloc[up_to_days:]
        qty_array = pd.Series([stock_quantities[t] for t in tickers], index=tickers)
        stock_pnl_series = stock_price_changes.multiply(qty_array, axis=1).sum(axis=1)
        return stock_pnl_series
    else:
        return pd.Series(dtype=float)

def get_option_portfolio_pnl_distribution(
    pf: portfolio.Portfolio,
    date: datetime.date,
    window_size: int,
    up_to_days: int,
) -> pd.Series:
    """
    Compute the distribution of PnL for the option positions in the portfolio on the given date using the last window_size days of historical price data.
    Computes price changes for each option and entire portfolio over up_to_days days.

    Args:
        pf: The portfolio to compute the PnL distribution for.
        date: The date on which to compute the PnL distribution.
        window_size: The number of days of historical price data to use for computing the PnL distribution.
        up_to_days: The number of days in the future to compute the PnL distribution for.

    Returns:
        A dictionary mapping dates to the corresponding PnL values.
    """

    # initialize option pnl series to 0 for all dates, then add option pnl values to it
    option_positions = pf.get_portfolio_option_positions_on_date(date)

    # using price changes over up_to_days days for underlying stock, so load prices with up_to_days included in the window size
    # ensures that we have the necessary price data to compute price changes over up_to_days days for the underlying stock, which is needed to compute the option price changes and option pnl for up_to_days days
    option_underlying_prices = risk.utils.load_underlying_stock_prices(pf, date, window_size, up_to_days)

    # get difference in underlying stock prices over up_to_days days for each option position to compute option price changes and option pnl for up_to_days days
    option_underlying_prices_diff = option_underlying_prices.diff(up_to_days).iloc[up_to_days:]

    if option_underlying_prices is not None:
        # get scenario start and end dates to get the difference in risk factors
        scenario_start_dates = option_underlying_prices.index[:-up_to_days]
        scenario_end_dates = option_underlying_prices.index[up_to_days:]
        
        # create pnl series initialized to 0 for all scenario end dates, then add option pnl values to it
        option_pnl_series = pd.Series(0.0, index=option_underlying_prices.index[up_to_days:])

    for pos in option_positions:
        if pos.underlying is None or pos.option_type is None or pos.strike is None or pos.expiry is None:
            raise ValueError("Option position must have underlying, option_type, strike, and expiry set.")

        # load underlying price shocks for each scenario on end date
        underlying_price_shocks = option_underlying_prices_diff[pos.underlying]

        # load risk free rate data for the up_to_days scenarios
        risk_free_rates_shocks = risk.utils.load_risk_free_rate_shocks(date, scenario_start_dates, scenario_end_dates)

        # load implied volatility for option position to compute option price changes over the up_to_days scenarios
        implied_vols_shocks = risk.utils.load_implied_vol_shocks(pos.underlying, pos.option_type, pos.strike, pos.expiry, scenario_start_dates, scenario_end_dates)

        # get time horizon to maturity of the option, reduce by up to days to consider decrease in expirys
        T_horizon = max((pos.expiry - date).days - up_to_days, 0) / 252 # convert to years assuming 252 trading days in a year

        # get option prices on current date with above shocks applied to compute option price changes and option pnl for up_to_days days
        option_prices = black_scholes.option_price(
            S=pos.get_underlying_price_on_date(date) + underlying_price_shocks,
            K=pos.strike,
            T=T_horizon, # reduced time to maturity by the horizon
            r=risk.utils.get_risk_free_rate(date) + risk_free_rates_shocks,
            sigma=pos.get_option_implied_vol_on_date(date) + implied_vols_shocks,
            option_type=pos.option_type
        )

        # get current option price to get price diffs
        curr_option_price = pos.get_option_price_on_date(date)

        option_pnl_series += pd.Series(
            (option_prices - curr_option_price) * pos.quantity,
            index=scenario_end_dates,
        )
    
    if option_underlying_prices is not None:
        return option_pnl_series
    else:
        return pd.Series(dtype=float)

def get_portfolio_pnl_distribution(
    pf: portfolio.Portfolio,
    date: datetime.date,
    window_size: int,
    up_to_days: int,
) -> dict[datetime.date, float]:
    """
    Compute the distribution of PnL for the portfolio on the given date using the last window_size days of historical price data.
    Computes price changes for each stock and entire portfolio over up_to_days days.

    Args:
        pf: The portfolio to compute the PnL distribution for.
        date: The date on which to compute the PnL distribution.
        window_size: The number of days of historical price data to use for computing the PnL distribution.
        up_to_days: The number of days in the future to compute the PnL distribution for.

    Returns:
        A dictionary mapping dates to the corresponding PnL values.
    """

    # load stock pnl distribution
    stock_pnl_series = get_stock_portfolio_pnl_distribution(pf, date, window_size, up_to_days)

    # load option pnl distribution
    option_pnl_series = get_option_portfolio_pnl_distribution(pf, date, window_size, up_to_days)

    # if both stock and option pnl data is available
    if not stock_pnl_series.empty and not option_pnl_series.empty:
        total_pnl_series = stock_pnl_series + option_pnl_series
        return total_pnl_series.to_dict()
    # if only stock pnl data is available
    elif not stock_pnl_series.empty and option_pnl_series.empty:
        total_pnl_series = stock_pnl_series
        return total_pnl_series.to_dict()
    # if only option pnl data is available
    elif stock_pnl_series.empty and not option_pnl_series.empty:
        total_pnl_series = option_pnl_series
        return total_pnl_series.to_dict()
    # if neither stock nor option pnl data is available, raise an error
    else:
        raise ValueError("Invalid combination of stock and option price/quantity data.")

def compute_portfolio_historical_var_es(
    pf: portfolio.Portfolio,
    date: datetime.date,
    window_size: int,
    up_to_days: int,
    alpha: float,
    lambda_val: float,
    method: str,
) -> tuple[list[float], float]:
    """
    Computes the historical VaR and ES of portfolio on the given date using the last window_size days of historical price data.
    VaR is the maximum loss that the portfolio could have experienced with a confidence level of alpha over the next up_to_days days.
    ES is the expected loss that the portfolio could have experienced with a confidence level of alpha over the next up_to_days days.
    Computes the VaR and ES by using up_to_days returns and not scaling the 1 day VaR/ES to up_to_days using the square root of time rule.

    Args:
        pf: The portfolio to compute VaR for.
        date: The date on which to compute the historical VaR.
        window_size: The number of days of historical price data to use for computing the historical VaR.
        up_to_days: The number of days in the future to compute the VaR for.
        alpha: The confidence level for the VaR calculation (e.g. 0.95 for 95% confidence level).
        lambda_val: The lambda parameter for weighted historical VaR.
        method: 1-day-scaled or n-day
    
    Returns:
        The historical VaR and ES of the portfolio on the given date.
    """

    if method == "n-day":
        pnl_data = get_portfolio_pnl_distribution(pf, date, window_size, up_to_days)
    elif method == "1-day-scaled":
        pnl_data = get_portfolio_pnl_distribution(pf, date, window_size, up_to_days=1)
    else:
        raise ValueError("Invalid method. Please use 'n-day' or '1-day-scaled'.")

    # compute the VaR at the given confidence level
    pnl_values = np.asarray(list(pnl_data.values()))

    # compute weights for weighted historical VaR and ES
    n = len(pnl_values)
    powers = np.arange(n - 1, -1, -1)
    weights = lambda_val ** powers
    weights /= weights.sum()

    idx = np.argsort(pnl_values)
    pnl_sorted = pnl_values[idx]
    weights_sorted = weights[idx]
    cum_weights = np.cumsum(weights_sorted)

    if method == "n-day":
        var = -np.interp(1 - alpha, cum_weights, pnl_sorted) # negative sign to convert from loss to positive VaR value
        mask = pnl_sorted <= -var
        tail_pnl = pnl_sorted[mask]
        tail_weights = weights_sorted[mask]
        tail_weights /= tail_weights.sum()
        es = -np.sum(tail_weights * tail_pnl) # negative sign to convert from loss to positive ES value
    elif method == "1-day-scaled":
        # compute 1-day VaR and ES first, then scale both to up_to_days using the square root of time rule
        var_1d = -np.interp(1 - alpha, cum_weights, pnl_sorted) # negative sign to convert from loss to positive VaR value
        mask = pnl_sorted <= -var_1d
        tail_pnl = pnl_sorted[mask]
        tail_weights = weights_sorted[mask]
        tail_weights /= tail_weights.sum()
        es_1d = -np.sum(tail_weights * tail_pnl) # negative sign to convert from loss to positive ES value

        var = var_1d * np.sqrt(up_to_days)
        es = es_1d * np.sqrt(up_to_days)
    else:
        raise ValueError("Invalid method. Please use 'n-day' or '1-day-scaled'.")

    return pnl_data, var, es
