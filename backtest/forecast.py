import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio import portfolio
from portfolio import position

from risk import historical, monte_carlo_sim, parametric
from risk import cache

def compute_portfolio_var_es_forecast(
    pf: portfolio.Portfolio,
    date: datetime.date,
    method: str,
    window_size: int,
    up_to_days: int,
    alpha: float,
    lambda_val: float,
    num_simulations: int | None = None,
    historical_method: str = "n-day",
    manual_return_moments: dict[str, dict[str, float]] | None = None,
) -> tuple[float, float]:
    """
    Compute a VaR and ES forecast for the portfolio on a given date using the specified risk method.
    This is a reusable wrapper around the existing risk engine functions and is intended to be
    called repeatedly by the backtesting framework on different forecast dates.

    Args:
        pf: The portfolio to forecast risk for.
        date: The forecast date on which to compute VaR and ES.
        method: The risk method to use. Must be one of "historical", "monte_carlo", or "parametric".
        window_size: The number of days of historical data to use in the forecast.
        up_to_days: The forecast horizon in days.
        alpha: The confidence level for the VaR/ES forecast.
        lambda_val: The decay parameter used in weighted calibration or weighted historical estimation.
        num_simulations: The number of Monte Carlo simulations to run when method is "monte_carlo".
        historical_method: The historical VaR methodology to use when method is "historical".
            Must be one of "n-day" or "1-day-scaled".
        manual_return_moments: Optional dictionary of user-specified return means
            and variances for each stock ticker.

    Returns:
        The VaR and ES forecast for the given date and method.
    """

    if method == "historical":
        _, var, es = historical.compute_portfolio_historical_var_es(
            pf,
            date,
            window_size,
            up_to_days,
            alpha,
            lambda_val,
            historical_method,
        )
    elif method == "monte_carlo":
        if num_simulations is None:
            raise ValueError("num_simulations must be provided when method is 'monte_carlo'.")

        _, var, es = monte_carlo_sim.compute_portfolio_monte_carlo_var_es(
            pf,
            date,
            window_size,
            up_to_days,
            alpha,
            lambda_val,
            num_simulations,
            manual_return_moments=manual_return_moments,
        )
    elif method == "parametric":
        var, es = parametric.compute_portfolio_parametric_var_es(
            pf,
            date,
            window_size,
            up_to_days,
            alpha,
            lambda_val,
            manual_return_moments=manual_return_moments,
        )
    else:
        raise ValueError("Invalid method. Please use 'historical', 'monte_carlo', or 'parametric'.")

    return var, es

def compute_portfolio_var_es_forecast_over_range(
    pf: portfolio.Portfolio,
    start_date: datetime.date,
    end_date: datetime.date,
    window_size: int,
    up_to_days: int,
    alpha: float,
    lambda_val: float,
    num_simulations: int | None = None,
    manual_return_moments: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Compute VaR and ES forecasts for the portfolio over a date range using the specified risk method.
    Calls the single-date forecast function on each valid forecast date and stores the results in a
    structured dataframe for use in backtesting and downstream analysis.

    Args:
        pf: The portfolio to forecast risk for.
        start_date: The first forecast date to include.
        end_date: The last forecast date to include.
        window_size: The number of days of historical data to use in each forecast.
        up_to_days: The forecast horizon in days.
        alpha: The confidence level for the VaR/ES forecast.
        lambda_val: The decay parameter used in weighted calibration or weighted historical estimation.
        num_simulations: The number of Monte Carlo simulations to run when method is "monte_carlo".
        manual_return_moments: Optional dictionary of user-specified return means
            and variances for each stock ticker.

    Returns:
        A pandas DataFrame containing one row per forecast date with shared forecast inputs,
        portfolio state information, and VaR/ES columns for each risk method.
    """

    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    # use the union of stock tickers and option underlyings across all 
    # positions to build a common forecast-date calendar for the portfolio
    # this prevents bad data issues when a forecast date is valid for some positions
    # but not others due to missing price history for certain tickers on certain dates
    tickers = sorted(
        set(
            pos.ticker if pos.tradable_type == "stock" else pos.underlying
            for pos in pf.positions
            if pos.ticker is not None
        )
    )

    if not tickers:
        return pd.DataFrame()

    aligned_dates = None
    for ticker in tickers:
        if ticker is None:
            continue

        price_history = cache.get_price_history(ticker)
        ticker_dates = set(price_history.index[(price_history.index >= start_date) & (price_history.index <= end_date)])

        if aligned_dates is None:
            aligned_dates = ticker_dates
        else:
            aligned_dates &= ticker_dates

    if not aligned_dates:
        return pd.DataFrame()

    forecast_dates = sorted(aligned_dates)
    total_forecast_dates = len(forecast_dates)
    forecast_rows = []
    forecast_methods = [
        {"method": "historical", "historical_method": "1-day-scaled", "num_simulations": None},
        {"method": "historical", "historical_method": "n-day", "num_simulations": None},
        {"method": "monte_carlo", "historical_method": None, "num_simulations": num_simulations},
        {"method": "parametric", "historical_method": None, "num_simulations": None},
    ]

    for i, date in enumerate(forecast_dates, start=1):
        if i % 100 == 0 or i == total_forecast_dates:
            progress_pct = 100 * i / total_forecast_dates
            print(
                f"[{i}/{total_forecast_dates}] "
                f"Computing forecasts for {date} "
                f"({progress_pct:.1f}% complete)"
            )

        active_positions = pf.get_positions_on_date(date)
        if not active_positions:
            continue

        forecast_row = {
            "date": date,
            "window_size": window_size,
            "up_to_days": up_to_days,
            "alpha": alpha,
            "lambda_val": lambda_val,
            "num_simulations": num_simulations,
            "horizon_end_date": date + datetime.timedelta(days=up_to_days),
            "num_active_positions": len(active_positions),
            "num_active_stock_positions": len(pf.get_portfolio_stock_positions_on_date(date)),
            "num_active_option_positions": len(pf.get_portfolio_option_positions_on_date(date)),
            "position_tickers": pf.get_position_tickers_on_date(date),
        }

        for forecast_method in forecast_methods:
            if forecast_method["method"] == "monte_carlo" and forecast_method["num_simulations"] is None:
                continue

            var, es = compute_portfolio_var_es_forecast(
                pf=pf,
                date=date,
                method=forecast_method["method"],
                window_size=window_size,
                up_to_days=up_to_days,
                alpha=alpha,
                lambda_val=lambda_val,
                num_simulations=forecast_method["num_simulations"],
                historical_method=forecast_method["historical_method"] or "n-day",
                manual_return_moments=manual_return_moments,
            )

            forecast_label = forecast_method["method"]
            if forecast_method["method"] == "historical":
                forecast_label = f"historical_{forecast_method['historical_method']}"

            forecast_row[f"{forecast_label}_var"] = var
            forecast_row[f"{forecast_label}_es"] = es

        forecast_rows.append(forecast_row)

    return pd.DataFrame(forecast_rows)
