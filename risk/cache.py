import os
import datetime
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

from pricing import black_scholes
from risk import utils

_price_history_cache: dict[str, pd.DataFrame] = {}
_risk_free_rate_history_cache: pd.DataFrame | None = None
_daily_price_surface_cache: dict[tuple[str, datetime.date], pd.DataFrame] = {}
_daily_vol_surface_cache: dict[tuple[str, datetime.date], pd.DataFrame] = {}

_maturity_vol_surface_cache: dict[tuple[str, datetime.date, datetime.date, str], pd.DataFrame] = {}
_maturity_vol_interpolator_cache: dict[tuple[str, datetime.date, datetime.date, str], LinearNDInterpolator] = {}

_fixed_expiry_vol_surface_cache: dict[tuple[str, datetime.date, str], pd.DataFrame] = {}
_fixed_expiry_vol_interpolator_cache: dict[tuple[str, datetime.date, str], LinearNDInterpolator] = {}

_implied_vol_point_cache: dict[
    # key: (mode, date, underlying, expiry, option_type, strike)
    # mode is either 'maturity' or 'fixed_expiry' to distinguish between the two types of interpolators
    # maturity is used in option pricing where we want to change expiry based on time to maturity
    # fixed_expiry is used in historical scenarios where we want to keep the expiry fixed

    # actually mode is always fixed_expiry for now
    # no caching for maturity-based interpolation
    tuple[str, datetime.date, str, datetime.date, str, float],
    float
] = {}

def _resolve_price_surface_path(ticker: str, date: datetime.date) -> str:
    """
    Resolve the daily price surface file for a ticker and date.
    Falls back to the latest available prior surface date when an exact file is not present.

    Args:
        ticker: The underlying ticker.
        date: The requested surface date.

    Returns:
        The resolved filesystem path to the daily price surface csv.
    """

    surface_dir = f"./data/vol_surfaces/{ticker}"
    exact_path = f"{surface_dir}/vol_surface_{date}.csv"
    if os.path.exists(exact_path):
        return exact_path

    if not os.path.isdir(surface_dir):
        raise ValueError(
            f"Price surface data directory for ticker {ticker} not found. Run the load_price_surface function to download the data."
        )

    prefix = "vol_surface_"
    suffix = ".csv"
    available_dates = []

    for filename in os.listdir(surface_dir):
        if not filename.startswith(prefix) or not filename.endswith(suffix):
            continue

        date_str = filename[len(prefix):-len(suffix)]
        try:
            file_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue

        if file_date <= date:
            available_dates.append(file_date)

    if not available_dates:
        raise ValueError(
            f"Price surface data for ticker {ticker} on or before date {date} not found. Run the load_price_surface function to download the data."
        )

    fallback_date = max(available_dates)
    return f"{surface_dir}/vol_surface_{fallback_date}.csv"

def _normalize_price_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price history data so it is indexed by Date using datetime.date values.

    Args:
        df: The raw price history dataframe loaded from csv.

    Returns:
        The normalized dataframe sorted and indexed by Date.
    """

    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
        df = df.sort_values("Date").set_index("Date")

    return df

def _normalize_daily_price_surface(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize daily option price surface data after reading from csv.

    Args:
        df: The raw daily option price surface dataframe.

    Returns:
        The normalized dataframe.
    """

    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.date

    return df

def clear_cache() -> None:
    _price_history_cache.clear()
    _daily_price_surface_cache.clear()
    _daily_vol_surface_cache.clear()
    _maturity_vol_surface_cache.clear()
    _maturity_vol_interpolator_cache.clear()
    _fixed_expiry_vol_surface_cache.clear()
    _fixed_expiry_vol_interpolator_cache.clear()
    _implied_vol_point_cache.clear()

    global _risk_free_rate_history_cache
    _risk_free_rate_history_cache = None

def get_price_history(ticker: str) -> pd.DataFrame:
    if ticker in _price_history_cache:
        return _price_history_cache[ticker]

    path = f"./data/prices/{ticker}.csv"
    if not os.path.exists(path):
        raise ValueError(f"Price data for ticker {ticker} not found. Run the load_prices function to download the data.")

    df = pd.read_csv(
        path,
        usecols=["Date", "Close"],
        parse_dates=["Date"],
    )
    df = _normalize_price_history(df)

    _price_history_cache[ticker] = df
    return df

def get_risk_free_rate_history() -> pd.DataFrame:
    global _risk_free_rate_history_cache
    if _risk_free_rate_history_cache is not None:
        return _risk_free_rate_history_cache

    path = "./data/treasury_yield/3m_treasury_yield.csv"
    if not os.path.exists(path):
        raise ValueError(f"3 month treasury yield data not found. Run the load_3m_treasury_yield function to download the data.")

    df = pd.read_csv(
        path,
        usecols=["Date", "Close"],
        parse_dates=["Date"],
    )
    df = _normalize_price_history(df)

    _risk_free_rate_history_cache = df
    return df

def get_daily_price_surface(ticker: str, date: datetime.date) -> pd.DataFrame:
    key = (ticker, date)
    if key in _daily_price_surface_cache:
        return _daily_price_surface_cache[key]

    path = _resolve_price_surface_path(ticker, date)

    df = pd.read_csv(
        path,
        usecols=["date", "symbol", "expiry", "type", "strike", "close"],
        parse_dates=["date"],
    )
    df = _normalize_daily_price_surface(df)

    _daily_price_surface_cache[key] = df
    return df

def get_daily_vol_surface(ticker: str, date: datetime.date) -> pd.DataFrame:
    key = (ticker, date)
    if key in _daily_vol_surface_cache:
        return _daily_vol_surface_cache[key]

    df = get_daily_price_surface(ticker, date).copy()
    df = black_scholes.extend_price_surface_with_implied_vol(df, date, ticker)

    _daily_vol_surface_cache[key] = df
    return df

def get_maturity_vol_interpolator(
    ticker: str,
    surface_date: datetime.date,
    valuation_date: datetime.date,
    option_type: str,
) -> tuple[pd.DataFrame, LinearNDInterpolator]:
    key = (ticker, surface_date, valuation_date, option_type)
    if key in _maturity_vol_interpolator_cache:
        return _maturity_vol_surface_cache[key], _maturity_vol_interpolator_cache[key]

    df = get_daily_vol_surface(ticker, surface_date).copy()
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True).dt.date
    df["time_to_maturity_in_days"] = [x.days for x in (df["expiry"] - valuation_date)]
    df["maturity"] = df["time_to_maturity_in_days"] / 252

    vol_surface = df[(df["symbol"].str.contains(ticker)) & (df["type"] == option_type)]
    vol_surface = vol_surface.dropna(subset=["strike", "maturity", "impliedVolatility"])

    interp_fn = LinearNDInterpolator(
        vol_surface[["strike", "maturity"]].values,
        vol_surface["impliedVolatility"].values,
    )

    _maturity_vol_surface_cache[key] = vol_surface
    _maturity_vol_interpolator_cache[key] = interp_fn
    return vol_surface, interp_fn

def get_fixed_expiry_vol_interpolator(
    ticker: str,
    surface_date: datetime.date,
    option_type: str,
) -> tuple[pd.DataFrame, LinearNDInterpolator]:
    key = (ticker, surface_date, option_type)
    if key in _fixed_expiry_vol_interpolator_cache:
        return _fixed_expiry_vol_surface_cache[key], _fixed_expiry_vol_interpolator_cache[key]

    df = get_daily_vol_surface(ticker, surface_date).copy()
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True).dt.date

    vol_surface = df[(df["symbol"].str.contains(ticker)) & (df["type"] == option_type)]
    vol_surface = vol_surface.dropna(subset=["strike", "expiry", "impliedVolatility"])
    vol_surface["expiry_ordinal"] = vol_surface["expiry"].map(datetime.date.toordinal)

    interp_fn = LinearNDInterpolator(
        vol_surface[["strike", "expiry_ordinal"]].values,
        vol_surface["impliedVolatility"].values,
    )

    _fixed_expiry_vol_surface_cache[key] = vol_surface
    _fixed_expiry_vol_interpolator_cache[key] = interp_fn
    return vol_surface, interp_fn

def get_fixed_expiry_implied_vol_point(
    surface_date: datetime.date,
    ticker: str,
    expiry: datetime.date,
    option_type: str,
    strike: float,
) -> float:
    """
    Get the implied vol point for a given date, ticker, expiry, option type and strike using the fixed expiry vol surface interpolator.
    
    Args:
        surface_date: The date for which to get the vol point.
        ticker: The underlying ticker.
        expiry: The option expiry date.
        option_type: The option type, either "call" or "put".
        strike: The option strike price.

    Returns:
        The implied vol point for the given parameters.
    """

    key = ("fixed_expiry", surface_date, ticker, expiry, option_type, strike)

    if key in _implied_vol_point_cache:
        return _implied_vol_point_cache[key]

    # get option implied volatility from cached price surface and interpolator
    vol_surface, interp_fn = get_fixed_expiry_vol_interpolator(ticker, surface_date, option_type)

    if vol_surface.empty:
        raise ValueError(f"No valid implied volatility points available for ticker {ticker} on {surface_date}.")
    
    if strike < vol_surface["strike"].min() or strike > vol_surface["strike"].max():
        raise ValueError(f"Strike price {strike} is out of bounds for the vol surface data for ticker {ticker}")

    expiries = np.sort(vol_surface["expiry"].unique())
    target_expiry_ordinal = expiry.toordinal()

    if len(expiries) == 1:
        expiry_slice_vol_surface = vol_surface[vol_surface["expiry"] == expiries[0]]
        implied_vol = utils.interpolate_vol_by_strike(expiry_slice_vol_surface, strike)
        _implied_vol_point_cache[key] = implied_vol
        return implied_vol

    # use nearest expiry slice if the requested expiry lies outside the observed expiry range
    if expiry < expiries.min() or expiry > expiries.max():
        warnings.warn(
            f"Expiry {expiry} is out of bounds for the vol surface data for ticker {ticker}; using nearest expiry for interpolation.",
            RuntimeWarning,
        )
        nearest_expiry = min(expiries, key=lambda x: abs((x - expiry).days))
        expiry_slice_vol_surface = vol_surface[vol_surface["expiry"] == nearest_expiry]
        implied_vol = utils.interpolate_vol_by_strike(expiry_slice_vol_surface, strike)
        _implied_vol_point_cache[key] = implied_vol
        return implied_vol

    implied_vol = interp_fn(strike, target_expiry_ordinal)

    if np.isnan(implied_vol):
        nearest_expiry = min(expiries, key=lambda x: abs((x - expiry).days))
        expiry_slice_vol_surface = vol_surface[vol_surface["expiry"] == nearest_expiry]
        implied_vol = utils.interpolate_vol_by_strike(expiry_slice_vol_surface, strike)
        _implied_vol_point_cache[key] = implied_vol
        return implied_vol

    _implied_vol_point_cache[key] = implied_vol
    return float(implied_vol)
