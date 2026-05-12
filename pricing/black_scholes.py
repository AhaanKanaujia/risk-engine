import os
import datetime
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

from portfolio import portfolio
from portfolio import position
from risk import utils, cache

def _build_valid_implied_vol_mask(S: float, K: np.ndarray, T: np.ndarray, market_price: np.ndarray, option_type: np.ndarray) -> np.ndarray:
    """
    Compute a mask indicating which option contracts have valid inputs for implied volatility inversion.

    Args:
        S: Current underlying stock price.
        K: Strike prices.
        T: Times to maturity in years.
        market_price: Observed market prices of the options.
        option_type: Option types ("call" or "put").

    Returns:
        A boolean numpy array indicating which rows are valid for implied volatility computation.
    """

    intrinsic_call = np.maximum(S - K, 0.0)
    intrinsic_put = np.maximum(K - S, 0.0)
    intrinsic = np.where(option_type == "call", intrinsic_call, intrinsic_put)

    return (
        np.isfinite(K) &
        np.isfinite(T) &
        np.isfinite(market_price) &
        (T > 0) &
        (market_price > intrinsic + 1e-8)
    )

def _newton_implied_vol(S: float, K: np.ndarray, T: np.ndarray, r: float, market_price: np.ndarray, option_type: np.ndarray, sigma0: np.ndarray, tol: float = 1e-8, max_iter: int = 20, sigma_min: float = 1e-6, sigma_max: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute implied volatilities using Newton's method with vectorized Black-Scholes price and vega calculations.

    Args:
        S: Current underlying stock price.
        K: Strike prices.
        T: Times to maturity in years.
        r: Risk-free interest rate.
        market_price: Observed market prices of the options.
        option_type: Option types ("call" or "put").
        sigma0: Initial guess for implied volatility.
        tol: Convergence tolerance.
        max_iter: Maximum number of Newton iterations.
        sigma_min: Lower bound for implied volatility.
        sigma_max: Upper bound for implied volatility.

    Returns:
        A tuple containing the implied volatilities and a boolean array indicating convergence.
    """

    sigma = sigma0.copy()
    converged = np.zeros(len(K), dtype=bool)

    for _ in range(max_iter):
        price = np.empty(len(K), dtype=float)
        vega = np.empty(len(K), dtype=float)

        call_mask = option_type == "call"
        put_mask = option_type == "put"

        if np.any(call_mask):
            price[call_mask] = option_price(S, K[call_mask], T[call_mask], r, sigma[call_mask], "call")
            vega[call_mask] = option_vega(S, K[call_mask], T[call_mask], r, sigma[call_mask])

        if np.any(put_mask):
            price[put_mask] = option_price(S, K[put_mask], T[put_mask], r, sigma[put_mask], "put")
            vega[put_mask] = option_vega(S, K[put_mask], T[put_mask], r, sigma[put_mask])

        error = price - market_price
        newly_converged = np.abs(error) < tol
        converged |= newly_converged

        active = (~converged) & (vega > 1e-10)
        if not np.any(active):
            break

        sigma[active] = sigma[active] - error[active] / vega[active]
        sigma[active] = np.clip(sigma[active], sigma_min, sigma_max)

    return sigma, converged

def _brent_implied_vol_fallback(S: float, K: np.ndarray, T: np.ndarray, r: float, market_price: np.ndarray, option_type: np.ndarray, sigma_min: float = 1e-6, sigma_max: float = 5.0) -> np.ndarray:
    """
    Compute implied volatilities using Brent's method for contracts where Newton's method does not converge.

    Args:
        S: Current underlying stock price.
        K: Strike prices.
        T: Times to maturity in years.
        r: Risk-free interest rate.
        market_price: Observed market prices of the options.
        option_type: Option types ("call" or "put").
        sigma_min: Lower bound for implied volatility.
        sigma_max: Upper bound for implied volatility.

    Returns:
        A numpy array of implied volatilities.
    """

    implied_vols = np.full(len(K), np.nan, dtype=float)

    for i in range(len(K)):
        def objective_function(sigma: float) -> float:
            return option_price(S, K[i], T[i], r, sigma, option_type[i]) - market_price[i]

        try:
            implied_vols[i] = optimize.brentq(objective_function, a=sigma_min, b=sigma_max)
        except ValueError:
            implied_vols[i] = np.nan

    return implied_vols

def infer_implied_vol_surface(S: float, K: np.ndarray, T: np.ndarray, r: float, market_price: np.ndarray, option_type: np.ndarray) -> np.ndarray:
    """
    Compute implied volatilities for an option price surface using a hybrid Newton and Brent root finding approach.

    Args:
        S: Current underlying stock price.
        K: Strike prices.
        T: Times to maturity in years.
        r: Risk-free interest rate.
        market_price: Observed market prices of the options.
        option_type: Option types ("call" or "put").

    Returns:
        A numpy array of implied volatilities for each contract in the price surface.
    """

    implied_vols = np.full(len(K), np.nan, dtype=float)
    valid_mask = _build_valid_implied_vol_mask(S, K, T, market_price, option_type)

    if not np.any(valid_mask):
        return implied_vols

    valid_idx = np.where(valid_mask)[0]
    sigma0 = np.full(len(valid_idx), 0.2, dtype=float)

    # first try Newton's method, which is faster when the initial guess is reasonable
    newton_sigma, converged = _newton_implied_vol(
        S=S,
        K=K[valid_idx],
        T=T[valid_idx],
        r=r,
        market_price=market_price[valid_idx],
        option_type=option_type[valid_idx],
        sigma0=sigma0,
    )

    implied_vols[valid_idx[converged]] = newton_sigma[converged]

    failed_idx = valid_idx[~converged]
    if len(failed_idx) > 0:
        # use Brent's method as a fallback for contracts where Newton's method fails to converge
        implied_vols[failed_idx] = _brent_implied_vol_fallback(
            S=S,
            K=K[failed_idx],
            T=T[failed_idx],
            r=r,
            market_price=market_price[failed_idx],
            option_type=option_type[failed_idx],
        )

    return implied_vols

def extend_price_surface_with_implied_vol(price_surface_df: pd.DataFrame, date: datetime.date, ticker: str) -> pd.DataFrame:
    """
    Extend price surface with implied vol computed by root finding method

    Args:
        price_surface_df: A dataframe containing the price surface data with columns "contractSymbol", "strike", "expiry", "option_type", and "lastPrice".
        date: The date for which to compute the implied volatility surface (always current date since we do not have historical proxy for implied volatility surface).
        ticker: The ticker for which to compute the implied volatility surface.

    Returns:
        A dataframe containing the price surface data with an additional column "impliedVolatility" containing the implied volatility for each option contract.
    """

    price_surface_df = price_surface_df.copy()
    price_surface_df["expiry"] = pd.to_datetime(price_surface_df["expiry"], utc=True).dt.date

    # these are constant across the entire daily price surface
    S = utils.load_underlying_stock_price_on_date(ticker, date)
    r = utils.get_risk_free_rate(date)

    K = price_surface_df["strike"].to_numpy(dtype=float)
    T = np.array([(expiry - date).days / 252 for expiry in price_surface_df["expiry"]], dtype=float)
    market_price = price_surface_df["close"].to_numpy(dtype=float)
    option_type = price_surface_df["type"].to_numpy()

    price_surface_df["impliedVolatility"] = infer_implied_vol_surface(
        S=S,
        K=K,
        T=T,
        r=r,
        market_price=market_price,
        option_type=option_type,
    )

    return price_surface_df

def _extrapolated_implied_vol(vol_surface: pd.DataFrame, strike: float, maturity: float) -> float:
    """
    Extrapolate implied volatility in the maturity direction using the two nearest boundary maturities.

    Args:
        vol_surface: A dataframe containing strike, maturity, and impliedVolatility columns.
        strike: The strike at which to compute implied volatility.
        maturity: The target maturity in years.

    Returns:
        The extrapolated implied volatility.
    """

    unique_maturities = np.sort(vol_surface["maturity"].unique())

    if len(unique_maturities) == 1:
        boundary_slice = vol_surface[vol_surface["maturity"] == unique_maturities[0]]
        return utils.interpolate_vol_by_strike(boundary_slice, strike)

    if maturity < unique_maturities.min():
        m1, m2 = unique_maturities[0], unique_maturities[1]
    else:
        m1, m2 = unique_maturities[-2], unique_maturities[-1]

    slice1 = vol_surface[vol_surface["maturity"] == m1]
    slice2 = vol_surface[vol_surface["maturity"] == m2]

    vol1 = utils.interpolate_vol_by_strike(slice1, strike)
    vol2 = utils.interpolate_vol_by_strike(slice2, strike)

    return float(vol1 + (maturity - m1) * (vol2 - vol1) / (m2 - m1))

def option_implied_vol(date: datetime.date, ticker: str, expiry: float, option_type: str, strike: float) -> float:
    """
    Read the implied volatility surface for a given ticker and expiry from the data folder where it is stored as a csv file.
    Looks for closest expiry in the vol surface data that is greater than or equal to the given expiry and returns the implied volatility surface for that expiry.
    Interpolates the implied volatility based on strike price if the exact strike price is not available in the vol surface data for the closest expiry.

    Args:
        date: The date for which to get the implied volatility surface (always current date since we do not have historical proxy for implied volatility surface).
        ticker: The ticker for which to get the implied volatility surface.
        expiry: The expiry date for which to get the implied volatility.
        option_type: The type of the option ("call" or "put").
        strike: The strike price for which to get the implied volatility.

    Returns:
        The implied volatility for the given strike price.
    """

    # floor the surface date to 2026-01-02 since we do not have historical proxy for implied volatility surface before that
    earliest_surface_date = datetime.date(2026, 1, 2)
    surface_date = max(date, earliest_surface_date)

    # get time to maturity in years for the given expiry
    time_to_maturity_in_days = (expiry - date).days
    time_to_maturity_in_years = time_to_maturity_in_days / 252 # convert to years assuming 252 trading days in a year

    # get option implied volatility from cached price surface and interpolator
    vol_surface, interp_fn = cache.get_maturity_vol_interpolator(ticker, surface_date, date, option_type)

    if vol_surface.empty:
        raise ValueError(f"No valid implied volatility points available for ticker {ticker} on {date}.")

    if strike < vol_surface["strike"].min() or strike > vol_surface["strike"].max():
        raise ValueError(f"Strike price {strike} is out of bounds for the vol surface data for ticker {ticker}")

    if time_to_maturity_in_years < vol_surface["maturity"].min() or time_to_maturity_in_years > vol_surface["maturity"].max():
        warnings.warn(
            f"Time to maturity {time_to_maturity_in_years} years is out of bounds for the vol surface data for ticker {ticker}; extrapolating implied volatility.",
            RuntimeWarning,
        )
        return _extrapolated_implied_vol(vol_surface, strike, time_to_maturity_in_years)
    
    implied_vol = interp_fn(strike, time_to_maturity_in_years)

    return implied_vol

def option_implied_vol_fixed_expiry(date: datetime.date, ticker: str, expiry: float, option_type: str, strike: float) -> float:
    """
    A variant of the option implied vol function that assumes fixed expiry instead of moving expiry.
    Used in historical implied vol shock calculations to avoid bias from moving to different expiry as we move through time.

    Args:
        date: The date for which to get the implied volatility surface (always current date since we do not have historical proxy for implied volatility surface).
        ticker: The ticker for which to get the implied volatility surface.
        expiry: The expiry date for which to get the implied volatility.
        option_type: The type of the option ("call" or "put").
        strike: The strike price for which to get the implied volatility.
    
    Returns:
        The implied volatility for the given strike price.
    """

    # floor the surface date to 2026-01-02 since we do not have historical proxy for implied volatility surface before that
    earliest_surface_date = datetime.date(2026, 1, 2)
    surface_date = max(date, earliest_surface_date)

    implied_vol = cache.get_fixed_expiry_implied_vol_point(
        surface_date=surface_date,
        ticker=ticker,
        expiry=expiry,
        option_type=option_type,
        strike=strike,
    )

    return implied_vol

def option_price(
    S: np.ndarray | list[float],
    K: float,
    T: np.ndarray | list[float],
    r: float | np.ndarray | list[float],
    sigma: float | np.ndarray | list[float],
    option_type: str,
) -> float:
    """
    Compute the Black-Scholes price of a European call or put option.
    Return intrinsic value if the option is expired (T <= 0).

    Args:
        S: Current price(s) of the underlying stock
        K: Strike price
        T: Time to maturity value(s) in years
        r: Risk-free rate(s)
        sigma: Volatility value(s)
        option_type: "call" or "put"

    Returns:
        The Black-Scholes price of the option. Returns a scalar for scalar inputs
        and a numpy array for vectorized inputs.
    """

    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Broadcast inputs so callers can pass any combination of scalars and vectors.
    S, T, r, sigma = np.broadcast_arrays(S, T, r, sigma)

    # compute d1/d2 for all entries (safe_T avoids divide-by-zero where T <= 0)
    safe_T = np.where(T > 0, T, 1.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * safe_T) / (sigma * np.sqrt(safe_T))
    d2 = d1 - sigma * np.sqrt(safe_T)

    if option_type == "call":
        live_price = S * norm.cdf(d1) - K * np.exp(-r * safe_T) * norm.cdf(d2)
        intrinsic  = np.maximum(S - K, 0)
    elif option_type == "put":
        live_price = K * np.exp(-r * safe_T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        intrinsic  = np.maximum(K - S, 0)
    else:
        raise ValueError(f"Invalid option type: {option_type}")
    
    prices = np.where(T > 0, live_price, intrinsic)

    if prices.ndim == 0:
        return float(prices)

    return prices

def option_delta(S: np.ndarray, K: float, T: np.ndarray, r: float, sigma: float, option_type: str) -> float:
    """
    Compute the Black-Scholes delta of a European call or put option.

    Args:
        S: Current price of the underlying stock (can be a numpy array for vectorized computation)
        K: Strike price
        T: Time to maturity in years (can be a numpy array for vectorized computation)
        r: Risk-free interest rate
        sigma: Volatility of the underlying stock
        option_type: "call" or "put"

    Returns:
        The Black-Scholes delta of the option.
    """

    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError(f"Invalid option type: {option_type}")
    
    return delta

def option_gamma(S: np.ndarray, K: float, T: np.ndarray, r: float, sigma: float) -> float:
    """
    Compute the Black-Scholes gamma of a European option.

    Args:
        S: Current price of the underlying stock (can be a numpy array for vectorized computation)
        K: Strike price
        T: Time to maturity in years (can be a numpy array for vectorized computation)
        r: Risk-free interest rate
        sigma: Volatility of the underlying stock
    
    Returns:
        The Black-Scholes gamma of the option.
    """

    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return gamma

def option_vega(S: np.ndarray, K: float, T: np.ndarray, r: float, sigma: float) -> float:
    """
    Compute the Black-Scholes vega of a European option.

    Args:
        S: Current price of the underlying stock (can be a numpy array for vectorized computation)
        K: Strike price
        T: Time to maturity in years (can be a numpy array for vectorized computation)
        r: Risk-free interest rate
        sigma: Volatility of the underlying stock
    
    Returns:
        The Black-Scholes vega of the option.
    """

    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return vega
