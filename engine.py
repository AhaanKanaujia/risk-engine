import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio import portfolio
from portfolio import position
from portfolio import data

from backtest import forecast, backtest, analysis

def main():
    # we need to download required data for the portfolio and forecast backtest
    # sample data uploaded to data folder, but may need to expand for larger portfolios

    # tickers = ["AAPL", "MSFT", "GOOGL"]
    # start_date = datetime.date(2026, 1, 1)
    # end_date = datetime.date.today()
    # prices = data.load_prices(tickers)
    # data.load_3m_treasury_yield()
    # vol_surfaces = data.load_volatility_surface(tickers)
    # for ticker in tickers:
        # data.fetch_vol_surface(ticker, str(start_date), str(end_date))


    # define your portfolio positions here
    start_date = datetime.date(2026, 1, 1)

    pf = portfolio.Portfolio()
    pf.add_position(position.Position(ticker="AAPL", quantity=100, tradable_type="stock", spot_price=250.0, date_opened=start_date))
    pf.add_position(position.Position(ticker="MSFT", quantity=50, tradable_type="stock", spot_price=475.0, date_opened=start_date))
    pf.add_position(position.Position(ticker="GOOGL", quantity=30, tradable_type="stock", spot_price=300.0, date_opened=start_date))
    # pf.add_position(position.Position(ticker="AAPL", quantity=1, tradable_type="stock", spot_price=250.0, date_opened=datetime.date(2026, 1, 1)))
    # pf.add_position(position.Position(ticker="MSFT", quantity=1, tradable_type="stock", spot_price=475.0, date_opened=datetime.date(2026, 1, 1)))
    # pf.add_position(position.Position(ticker="GOOGL", quantity=1, tradable_type="stock", spot_price=300.0, date_opened=datetime.date(2026, 1, 1)))
    
    pf.add_position(position.Position(ticker="AAPL_C", quantity=100, tradable_type="option", spot_price=40.0, option_type="call", strike=250.0, expiry=datetime.date(2026, 6, 27), underlying="AAPL", date_opened=datetime.date(2026, 1, 1)))
    pf.add_position(position.Position(ticker="MSFT_P", quantity=100, tradable_type="option", spot_price=30.0, option_type="put", strike=475.0, expiry=datetime.date(2026, 6, 27), underlying="MSFT", date_opened=datetime.date(2026, 1, 1)))
    # pf.add_position(position.Position(ticker="AAPL_C", quantity=1, tradable_type="option", spot_price=40.0, option_type="call", strike=250.0, expiry=datetime.date(2026, 2, 27), underlying="AAPL", date_opened=datetime.date(2026, 1, 1)))
    # pf.add_position(position.Position(ticker="MSFT_P", quantity=1, tradable_type="option", spot_price=30.0, option_type="put", strike=475.0, expiry=datetime.date(2026, 2, 27), underlying="MSFT", date_opened=datetime.date(2026, 1, 1)))

    start_date = start_date
    end_date = datetime.date(2026, 4, 21)

    # please do not go back more than 100 days for portfolios with option positions
    # vol surface data is hard to find, so defaults to 2026-01-02 for all dates before that
    # also takes a very long time to calculate implied vols from price surface
    # finally with weights approx. 0.95, prior to 100 days is insignificant for calibration
    window_size = 100

    # keep up to days less than a month, beyond that we're modelling noise
    up_to_days = 10

    # standard confidence level for VaR and ES calculations
    # can experiment with different confidence levels to see how it affects the VaR and ES values
    alpha = 0.95

    # lambda parameter for the calibration function, which controls the weighting of historical data in the calibration process
    # can experiment with different lambda values to see how it affects the calibration results and the resulting VaR and ES values
    # usually set to 0.9, 0.95, 0.975 or 0.99; set to 1.0 for equal weighting of all historical data
    lambda_val = 0.99

    # optional manual return moments for stocks in the portfolio
    # set to None to use historical calibration
    # otherwise provide mean and variance of daily returns for every stock ticker
    # manual_return_moments = None
    manual_return_moments = {
        "AAPL": {"mean": 0.0005, "variance": 0.0004},
        "MSFT": {"mean": 0.0004, "variance": 0.0003},
        "GOOGL": {"mean": 0.00045, "variance": 0.00035},
    }

    # 100k simulations is usually sufficient for stable monte carlo var and es calculations
    # increasing this will likely make the code slower with marginal to no improvements in accuracy
    num_simulations = 100_000

    forecast_df = forecast.compute_portfolio_var_es_forecast_over_range(
        pf=pf,
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        up_to_days=up_to_days,
        alpha=alpha,
        lambda_val=lambda_val,
        num_simulations=num_simulations,
        manual_return_moments=manual_return_moments,
    )

    realized_forecast_pnl_df = backtest.extend_results_with_realized_pnl(
        results_df=forecast_df,
        pf=pf,
    )

    backtested_pnl_df = backtest.backtest_exceeded_var_es_portfolio(
        results_df=realized_forecast_pnl_df,
    )

    backtest_summary_df = analysis.summarize_backtest_results(backtested_pnl_df)
    windows_plot = analysis.plot_backtest_exceedance_windows(backtested_pnl_df)
    scatter_plot = analysis.plot_backtest_exceedance_scatter(backtested_pnl_df)
    hit_sequence_plot = analysis.plot_backtest_hit_sequence(backtested_pnl_df)
    loss_vs_threshold_plot = analysis.plot_backtest_realized_loss_vs_threshold(backtested_pnl_df)

    print("Done with main execution flow.")

if __name__ == "__main__":
    main()
