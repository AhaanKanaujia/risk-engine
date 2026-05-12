import datetime
import numpy as np
import pandas as pd

from risk import cache
from portfolio import portfolio
from pricing import black_scholes

# helper function to interpolate implied vol across strike for a fixed maturity slice of the vol surface
# used in implied vol shocks computation for historical Var/ES calculations
# put here becuase cache and risk both need to use this function and we want to avoid circular imports
def interpolate_vol_by_strike(surface_slice: pd.DataFrame, strike: float) -> float:
    """
    Interpolate implied volatility across strike for a fixed maturity slice.

    Args:
        surface_slice: A dataframe containing rows for one maturity.
        strike: The strike at which to interpolate implied volatility.

    Returns:
        The implied volatility interpolated across strike.
    """

    surface_slice = surface_slice.sort_values("strike").drop_duplicates(subset=["strike"], keep="last")
    strikes = surface_slice["strike"].to_numpy(dtype=float)
    implied_vols = surface_slice["impliedVolatility"].to_numpy(dtype=float)

    return float(np.interp(strike, strikes, implied_vols))

# currently not being used, kept in case methodology changes
# loads underlying prices earlier to construct option pnl series
def load_start_end_underlying_prices(underlying: str, date: datetime.date, scenario_start_dates: pd.DatetimeIndex, scenario_end_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and align underlying prices for the given underlying asset over the specified scenario dates.
    Called in option pnl computation part of historical Var/ES calculations.

    Args:
        underlying: The ticker of the underlying asset.
        date: The current date for which to load the underlying prices.
        scenario_start_dates: The start dates of the historical scenarios.
        scenario_end_dates: The end dates of the historical scenarios.
    
    Returns:
        A tuple containing:
        - A numpy array of the underlying prices on the scenario start dates.
        - A numpy array of the underlying prices on the scenario end dates.
    """

    underlying_df = cache.get_price_history(underlying)
    underlying_prices = underlying_df.loc[underlying_df.index <= date, ["Close"]]

    underlying_prices_start = underlying_prices.reindex(scenario_start_dates)["Close"].to_numpy()
    underlying_prices_end = underlying_prices.reindex(scenario_end_dates)["Close"].to_numpy()

    if np.isnan(underlying_prices_start).any() or np.isnan(underlying_prices_end).any():
        raise ValueError(
            f"Missing aligned underlying prices for underlying {underlying} over the historical scenario window."
        )

    return underlying_prices_start, underlying_prices_end

def load_underlying_stock_price_on_date(ticker: str, date: datetime.date) -> float:
    """
    Load the underlying stock price for a given ticker on a given date from the data folder where it is stored as a csv file.

    Args:
        ticker: The ticker of the underlying stock.
        date: The date on which to load the underlying stock price.

    Returns:

        The underlying stock price on the given date.
    """

    underlying_df = cache.get_price_history(ticker)
    underlying_price = underlying_df.loc[underlying_df.index == date, ["Close"]]

    if underlying_price.empty:
        raise ValueError(
            f"Price data for underlying ticker {ticker} not found for date {date}."
        )

    return float(underlying_price["Close"].iloc[0])

def load_risk_free_rate_shocks(date: datetime.date, scenario_start_dates: pd.DatetimeIndex, scenario_end_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Load and align risk free rate shocks for the specified scenario dates.
    Called in option pnl computation part of historical Var/ES calculations.

    Args:
        date: The current date for which to load the risk free rates.
        scenario_start_dates: The start dates of the historical scenarios.
        scenario_end_dates: The end dates of the historical scenarios.
    
    Returns:
        A numpy array of the risk free rate shocks for the historical scenarios.
    """

    risk_free_df = cache.get_risk_free_rate_history()
    risk_free_rate = risk_free_df.loc[risk_free_df.index <= date, ["Close"]]

    risk_free_rates_start = risk_free_rate.reindex(scenario_start_dates)["Close"].to_numpy() / 100
    risk_free_rates_end = risk_free_rate.reindex(scenario_end_dates)["Close"].to_numpy() / 100

    if np.isnan(risk_free_rates_start).any() or np.isnan(risk_free_rates_end).any():
        raise ValueError(
            f"Missing aligned risk free rates over the historical scenario window."
        )

    return risk_free_rates_end - risk_free_rates_start

def load_implied_vol_shocks(underlying: str, option_type: str, strike: float, expiry: datetime.date, scenario_start_dates: pd.DatetimeIndex, scenario_end_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Load and align implied vol schocks for the given option parameters over the specified scenario dates.
    Does not take into account change in time to expiry, only looks at change in implied vol for the same expiry, strike, and option type.
    Called in option pnl computation part of historical Var/ES calculations.

    Args:
        underlying: The ticker of the underlying asset.
        option_type: The type of the option ("call" or "put").
        strike: The strike price of the option.
        expiry: The expiry date of the option.
        scenario_start_dates: The start dates of the historical scenarios.
        scenario_end_dates: The end dates of the historical scenarios.
    
    Returns:
        A numpy array of the implied vol shocks for the historical scenarios.
    """

    unique_dates = sorted(set(scenario_start_dates) | set(scenario_end_dates))

    # get implied vol for each unique date in scenario dates and cache the results to avoid redundant calculations
    implied_vols_by_date = {
        d: black_scholes.option_implied_vol_fixed_expiry(d, underlying, expiry, option_type, strike)
        for d in unique_dates
    }

    implied_vols_start = np.array([implied_vols_by_date[d] for d in scenario_start_dates])
    implied_vols_end = np.array([implied_vols_by_date[d] for d in scenario_end_dates])

    return implied_vols_end - implied_vols_start

def load_start_end_times_to_expiry(expiry: datetime.date, scenario_start_dates: pd.DatetimeIndex, scenario_end_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and align times to expiry for the given option expiry date over the specified scenario dates.
    Called in option pnl computation part of historical Var/ES calculations.

    Args:
        expiry: The expiry date of the option.
        scenario_start_dates: The start dates of the historical scenarios.
        scenario_end_dates: The end dates of the historical scenarios.
    
    Returns:
        A tuple containing:
        - A numpy array of the times to expiry on the scenario start dates.
        - A numpy array of the times to expiry on the scenario end dates.
    """

    times_to_expiry_start = np.array([(expiry - start_date).days / 252 for start_date in scenario_start_dates])
    times_to_expiry_end = np.array([(expiry - end_date).days / 252 for end_date in scenario_end_dates])

    return times_to_expiry_start, times_to_expiry_end

def load_underlying_stock_prices(pf: portfolio.Portfolio, date: datetime.date, window_size: int, up_to_days: int) -> pd.DataFrame:
    """
    Load historical Close prices for all underlying stock of an option position, indexed by Date, and aligned to ensure we have the necessary price data to compute price changes over up_to_days days.

    Args:
        pf: The portfolio containing the option positions to determine the underlying stocks we need to load price data for.
        date: The current date for which to load the underlying stock prices.
        window_size: The number of days of historical price data to use.
        up_to_days: The number of days into the future for which to compute price changes.
    
    Returns:
        A pandas DataFrame containing the aligned historical Close prices for the underlying stocks, indexed by Date.
    """

    # get all option positions in the portfolio on the given date to determine the underlying stocks we need to load price data for
    option_positions = pf.get_portfolio_option_positions_on_date(date)
    
    # load and align historical Close prices for the underlying stocks by Date
    underlying_price_frames = []
    loaded_underlyings = set()
    for pos in option_positions:
        if pos.underlying is None:
            raise ValueError("Option position must have underlying field set.")

        underlying = pos.underlying
        # prevent loading the same underlying price data multiple times if multiple option positions share the same underlying
        if underlying in loaded_underlyings:
            continue
        loaded_underlyings.add(underlying)

        df = cache.get_price_history(underlying)
        df = df.loc[df.index <= date, ["Close"]].rename(columns={"Close": underlying})
        underlying_price_frames.append(df)

    # if no price data available for any of the underlying stocks, return empty results
    if not underlying_price_frames:
        return pd.Series(dtype=float)

    # join on Date and keep only rows where all underlying stocks have data
    underlying_prices = pd.concat(underlying_price_frames, axis=1).sort_index().dropna()

    # need window_size + up_to_days rows to compute window_size daily differences
    if len(underlying_prices) < window_size + up_to_days:
        raise ValueError(f"Not enough aligned historical data for underlying stocks: have {len(underlying_prices)} rows, need {window_size + up_to_days}.")

    underlying_prices = underlying_prices.tail(window_size + up_to_days)

    return underlying_prices

def load_stock_prices_quantities(pf: portfolio.Portfolio, date: datetime.date, window_size: int, up_to_days: int) -> tuple[list[str], pd.DataFrame, dict[str, int]]:
    """
    Load and align historical Close prices for all tickers in the portfolio on the given date, along with their corresponding quantities.
    
    Args:
        pf: The portfolio to load prices and quantities for.
        date: The date on which to load the prices and quantities.
        window_size: The number of days of historical price data to use.
        up_to_days: The number of days into the future for which to compute PnL.

    Returns:
        A tuple containing:
        - A list of tickers in the portfolio.
        - A pandas DataFrame containing the aligned historical Close prices for all tickers, indexed by Date.
        - A dictionary mapping each ticker to its corresponding quantity in the portfolio on the given date
    """

    # get the positions in the portfolio on the given date
    positions = pf.get_portfolio_stock_positions_on_date(date)

    # determine tickers and aggregate quantities per ticker (in case of multiple positions same ticker)
    tickers = sorted(set(pos.ticker for pos in positions))
    quantities = pf.get_portfolio_stock_quantities_on_date(date)

    # load and align historical Close prices for all tickers by Date
    price_frames = []
    for ticker in tickers:
        df = cache.get_price_history(ticker)
        df = df.loc[df.index <= date, ["Close"]].rename(columns={"Close": ticker})
        price_frames.append(df)

    # if no price data available for any of the tickers, return empty results
    if not price_frames:
        return [], pd.DataFrame(), {}
    
    # join on Date and keep only rows where all tickers have data
    prices = pd.concat(price_frames, axis=1).sort_index().dropna()

    # need window_size + up_to_days rows to compute window_size daily differences
    if len(prices) < window_size + up_to_days:
        raise ValueError(f"Not enough aligned historical data across tickers: have {len(prices)} rows, need {window_size + up_to_days}.")

    prices = prices.tail(window_size + up_to_days)

    return tickers, prices, quantities

def get_risk_free_rate(date: datetime.date) -> float:
    """
    Load the risk-free rate on a given date from the data folder where it is stored as a csv file.

    Args:
        date: The date on which to load the risk-free rate.

    Returns:
        The risk-free rate on the given date.
    """

    df = cache.get_risk_free_rate_history()
    yield_data = df.loc[df.index <= date].tail(1)

    if yield_data.empty:
        raise ValueError(f"No 3 month treasury yield data available on or before date {date}.")
    
    return yield_data.iloc[0]["Close"] / 100 # convert from percentage to decimal
