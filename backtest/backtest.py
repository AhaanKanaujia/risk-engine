import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio import portfolio
from portfolio import position
from portfolio import data
from risk import cache

def _resolve_realized_end_date(
    positions: list[position.Position],
    requested_end_date: datetime.date,
) -> datetime.date | None:
    """
    Resolve a common realized end date for a set of portfolio positions.
    Uses the latest date on or before the requested end date for which all relevant
    underlying/stock price histories have data.

    Args:
        positions: The positions active on the forecast date.
        requested_end_date: The requested forward horizon end date.

    Returns:
        The resolved common realized end date, or None if no common date exists.
    """

    relevant_tickers = sorted(
        set(
            pos.ticker if pos.tradable_type == "stock" else pos.underlying
            for pos in positions
            if (pos.ticker if pos.tradable_type == "stock" else pos.underlying) is not None
        )
    )

    if not relevant_tickers:
        return None

    common_dates = None
    for ticker in relevant_tickers:
        price_history = cache.get_price_history(ticker)
        ticker_dates = set(price_history.index[price_history.index <= requested_end_date])

        if common_dates is None:
            common_dates = ticker_dates
        else:
            common_dates &= ticker_dates

    if not common_dates:
        return None

    return max(common_dates)

def _compute_option_realized_end_value(
    pos: position.Position,
    realized_end_date: datetime.date,
) -> float:
    """
    Compute the realized end value of an option position over the backtest horizon.
    If the option has expired by the realized end date, value it at intrinsic payoff at expiry.

    Args:
        pos: The option position to value.
        realized_end_date: The common realized horizon end date.

    Returns:
        The realized end value of the option contract.
    """

    if pos.expiry is None or pos.option_type is None or pos.strike is None:
        raise ValueError("Option position must have expiry, option_type, and strike fields set.")

    if realized_end_date < pos.expiry:
        return pos.get_option_price_on_date(realized_end_date)

    underlying_price_at_expiry = pos.get_underlying_price_on_date(pos.expiry)

    if pos.option_type == "call":
        return max(underlying_price_at_expiry - pos.strike, 0.0)
    if pos.option_type == "put":
        return max(pos.strike - underlying_price_at_expiry, 0.0)

    raise ValueError(f"Invalid option type: {pos.option_type}")

def _compute_portfolio_realized_pnl(
    pf: portfolio.Portfolio,
    forecast_date: datetime.date,
    requested_end_date: datetime.date,
) -> tuple[datetime.date | None, float]:
    """
    Compute realized portfolio PnL from the forecast date to the resolved forward horizon date.
    The portfolio is frozen at the forecast date and the same positions are repriced at the end date.

    Args:
        pf: The portfolio to backtest.
        forecast_date: The forecast origin date.
        requested_end_date: The requested horizon end date.

    Returns:
        A tuple containing the resolved realized end date and the realized portfolio PnL.
    """

    active_positions = (
        pf.get_portfolio_stock_positions_on_date(forecast_date)
        + pf.get_portfolio_option_positions_on_date(forecast_date)
    )
    if not active_positions:
        return None, np.nan

    realized_end_date = _resolve_realized_end_date(active_positions, requested_end_date)
    if realized_end_date is None:
        return None, np.nan

    realized_pnl = 0.0

    for pos in active_positions:
        if pos.tradable_type == "stock":
            start_value = pos.get_stock_price_on_date(forecast_date)
            end_value = pos.get_stock_price_on_date(realized_end_date)
        elif pos.tradable_type == "option":
            start_value = pos.get_option_price_on_date(forecast_date)
            end_value = _compute_option_realized_end_value(pos, realized_end_date)
        else:
            raise ValueError(f"Invalid tradable type: {pos.tradable_type}")

        realized_pnl += (end_value - start_value) * pos.quantity

    return realized_end_date, realized_pnl

def extend_results_with_realized_pnl(
    results_df: pd.DataFrame,
    pf: portfolio.Portfolio,
) -> pd.DataFrame:
    """
    Extend a VaR/ES forecast results dataframe with realized forward PnL over the specified horizon.
    The realized PnL is computed for the portfolio held on each forecast date and repriced over
    the row's requested up_to_days horizon.

    Args:
        results_df: The forecast results dataframe produced by the backtest forecast module.
        pf: The portfolio to compute realized forward PnL for.

    Returns:
        A copy of the input dataframe with realized end date and realized PnL columns added.
    """

    if results_df.empty:
        return results_df.copy()

    required_columns = {"date", "up_to_days"}
    missing_columns = required_columns - set(results_df.columns)
    if missing_columns:
        raise ValueError(
            f"results_df must contain the following columns: {sorted(required_columns)}. "
            f"Missing columns: {sorted(missing_columns)}."
        )

    extended_df = results_df.copy()
    realized_end_dates = []
    realized_pnls = []

    for _, row in extended_df.iterrows():
        forecast_date = row["date"]
        if isinstance(forecast_date, pd.Timestamp):
            forecast_date = forecast_date.date()

        requested_end_date = row.get("horizon_end_date")
        if requested_end_date is None or pd.isna(requested_end_date):
            requested_end_date = forecast_date + datetime.timedelta(days=int(row["up_to_days"]))
        elif isinstance(requested_end_date, pd.Timestamp):
            requested_end_date = requested_end_date.date()

        realized_end_date, realized_pnl = _compute_portfolio_realized_pnl(
            pf=pf,
            forecast_date=forecast_date,
            requested_end_date=requested_end_date,
        )

        realized_end_dates.append(realized_end_date)
        realized_pnls.append(realized_pnl)

    extended_df["realized_end_date"] = realized_end_dates
    extended_df["realized_pnl"] = realized_pnls

    return extended_df

def backtest_exceeded_var_es_portfolio(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extend a backtest results dataframe with exceedance flags and exceedance amounts
    for each VaR and ES forecast column.

    Args:
        results_df: A dataframe containing realized_pnl and forecast VaR/ES columns.

    Returns:
        A copy of the input dataframe with additional exceedance columns for each method-risk-measure combination.
    """

    if results_df.empty:
        return results_df.copy()

    if "realized_pnl" not in results_df.columns:
        raise ValueError("results_df must contain a realized_pnl column before exceedance backtesting.")

    extended_df = results_df.copy()
    forecast_columns = [
        col for col in extended_df.columns
        if col.endswith("_var") or col.endswith("_es")
    ]

    for forecast_col in forecast_columns:
        threshold = extended_df[forecast_col]
        realized_loss = -extended_df["realized_pnl"]

        exceeded_mask = (
            extended_df["realized_pnl"].notna()
            & threshold.notna()
            & (realized_loss > threshold)
        )

        exceed_amount = np.where(
            exceeded_mask,
            realized_loss - threshold,
            0.0,
        )

        extended_df[f"{forecast_col}_exceeded"] = exceeded_mask.astype(int)
        extended_df[f"{forecast_col}_exceed_amount"] = exceed_amount

    return extended_df
