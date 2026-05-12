# Risk Engine

Research-oriented Python risk engine for equity and option portfolios, with historical, Monte Carlo, and parametric VaR/ES forecasting and backtesting.

## What this project does

This repository implements an end-to-end portfolio risk workflow for:

- equity-only portfolios
- options-only portfolios
- mixed equity-option portfolios

The engine:

- values stocks from historical price data
- reprices European options with Black-Scholes
- constructs implied-volatility surfaces from option prices
- forecasts portfolio VaR and ES with multiple methodologies
- backtests those forecasts against realized forward PnL
- produces statistical summaries and diagnostic plots

## Implemented risk methodologies

The project compares four forecast series:

- `historical_1-day-scaled`
- `historical_n-day`
- `monte_carlo`
- `parametric`

For each method, the engine computes:

- Value at Risk (VaR)
- Expected Shortfall (ES)

## Repository structure

```text
backtest/     Forecast generation, realized PnL extension, backtests, analysis, tests
data/         Sample historical prices, rates, and option-surface files
models/       Calibration and model helpers
portfolio/    Portfolio, position, and market-data loading logic
pricing/      Black-Scholes pricing and implied-volatility routines
reporting/    Scenario libraries and experiment runners
risk/         Historical, Monte Carlo, parametric, cache, and utilities
results/      Saved experiment outputs and plots
engine.py     Example end-to-end workflow on a custom portfolio
main.py       Batch runner for configured reporting experiments
```

## Data requirements

The repository includes sample data so the project can be run directly:

- historical equity price data
- Treasury-yield-based risk-free rate data
- sample option-surface data

Options-heavy experiments depend on the available implied-volatility surface history. In the current dataset, the option-surface coverage is concentrated in the local 2026 window, so long historical stock backtests are richer than long historical option backtests.

## Local setup

Clone the repository and create a local Python environment:

```bash
git clone <your-repo-url>
cd risk-engine
python3 -m venv risk-engine-env
source risk-engine-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file contains the core runtime dependencies used by the project.

If you want to fetch additional option-surface data, create a local `.env` file with Alpaca credentials:

```env
API_KEY=your_alpaca_key
API_SECRET=your_alpaca_secret
```

The `.env` file is ignored by git.

If you only want to run the project using the sample data already included in `data/`, you can skip the API credentials step.

## Running the project

After installation, the quickest way to verify the repository is working is:

```bash
python3 main.py
```

This runs the configured reporting scenarios against the sample data and saves outputs under `results/`.

### 1. Run the configured reporting experiments

`main.py` runs the predefined scenario set in `reporting/experiment_configs.py`:

```bash
python3 main.py
```

This creates saved outputs under `results/`, grouped by:

- `equity_only`
- `options_only`
- `equity_option_combinations`

Each scenario folder contains:

- `forecast_results.csv`
- `realized_results.csv`
- `backtested_results.csv`
- `summary.csv`
- diagnostic plots

### 2. Run a custom portfolio workflow

`engine.py` shows the full workflow for a manually specified portfolio:

```bash
python3 engine.py
```

That script demonstrates:

- building a portfolio from stock and option positions
- choosing a date range and hyperparameters
- generating VaR/ES forecasts
- extending results with realized PnL
- backtesting exceedances
- creating tables and plots

## Reporting framework

The `reporting/` package is the easiest way to run structured experiments.

- `portfolio_library.py`
  - reusable portfolio builders
- `experiment_configs.py`
  - named scenarios, date ranges, and hyperparameters
- `run_experiment.py`
  - runs one scenario end to end
- `run_all.py`
  - runs all configured scenarios

By default, reporting skips a scenario if the expected output files already exist. To force recomputation, use `force_rerun=True`.

## Backtesting outputs

The summary and plots include:

- expected vs. actual exceedance counts
- Kupiec proportion-of-failures test
- exact binomial coverage test
- Basel-style traffic-light classification for VaR
- rolling exceedance-count plots
- exceedance hit-sequence plots
- realized-loss-versus-threshold plots
- exceedance scatter plots

## Notes and current limitations

- The engine is designed as a research prototype, not a production trading system.
- VaR backtesting is the primary formal validation target; ES is reported mainly as a complementary tail-severity measure.
- Historical simulation for option portfolios depends heavily on the quality and coverage of the implied-volatility surface data.
- Parametric and Monte Carlo methods use simplifying assumptions and may be less reliable for strongly nonlinear option portfolios.

## Suggested workflow for new users

1. Run `python3 main.py` to generate the default experiment set.
2. Inspect the saved `summary.csv` files and plots under `results/`.
3. Open `reporting/portfolio_library.py` and `reporting/experiment_configs.py` to define new scenarios.
4. Use `engine.py` if you want to test a custom portfolio directly.

## Project outcome

The project is most useful as:

- a derivatives-risk research framework
- a portfolio VaR/ES comparison engine
- a backtesting and diagnostic environment for linear and nonlinear portfolios

It is especially well suited for studying where historical, Monte Carlo, and parametric risk measures agree, where they diverge, and how their calibration changes across different portfolio structures and market regimes.
