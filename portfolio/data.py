import os
import csv
import time
import requests
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


def _load_env_file() -> None:
    """Populate missing environment variables from a repo-local .env file."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)

_load_env_file()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Set API_KEY and API_SECRET in the environment or .env file.")
 
TRADING_BASE = "https://paper-api.alpaca.markets"
DATA_BASE    = "https://data.alpaca.markets"
 
HEADERS = {
    "accept":              "application/json",
    "APCA-API-KEY-ID":     API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
}

def _trading_days(start: str, end: str) -> list[str]:
    days, current, last = [], datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    while current <= last:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += datetime.timedelta(days=1)
    return days
 
def _get_contracts(
    symbol: str,
    expiry_gte: str,
    expiry_lte: Optional[str] = None,
    status: Optional[str] = "active",
) -> list[str]:
    url = f"{TRADING_BASE}/v2/options/contracts"
    params = {
        "underlying_symbols":  symbol.upper(),
        "expiration_date_gte": expiry_gte,
        "limit":               10000,
    }
    if status is not None:
        params["status"] = status
    if expiry_lte is not None:
        params["expiration_date_lte"] = expiry_lte
    symbols = []
    while True:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        contracts = data if isinstance(data, list) else data.get("option_contracts", [])
        for c in contracts:
            symbols.append(c["symbol"])
        next_token = data.get("next_page_token") if isinstance(data, dict) else None
        if not next_token:
            break
        params["page_token"] = next_token
    return symbols

def _get_contracts_for_day(symbol: str, day: str, expiry_lte: Optional[str] = None) -> list[str]:
    """
    Fetch both active and inactive contracts that would be relevant for a
    historical observation day.

    Args:
        symbol: Underlying ticker symbol.
        day: Observation date as YYYY-MM-DD.
        expiry_lte: Optional upper expiry bound.

    Returns:
        A sorted unique list of contract symbols.
    """

    active_contracts = _get_contracts(symbol, expiry_gte=day, expiry_lte=expiry_lte, status="active")
    inactive_contracts = _get_contracts(symbol, expiry_gte=day, expiry_lte=expiry_lte, status="inactive")

    return sorted(set(active_contracts).union(inactive_contracts))
 
def _get_bars_for_day(symbols: list[str], day: str) -> dict:
    url, start, end = f"{DATA_BASE}/v1beta1/options/bars", f"{day}T00:00:00Z", f"{day}T23:59:59Z"
    all_bars = {}
    for i in range(0, len(symbols), 100):
        batch_symbols = symbols[i:i + 100]
        params = {
            "symbols":   ",".join(batch_symbols),
            "timeframe": "1Day",
            "start":     start,
            "end":       end,
            "limit":     10000,
            "sort":      "asc",
        }
        try:
            while True:
                resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                for sym, bars in data.get("bars", {}).items():
                    all_bars.setdefault(sym, []).extend(bars)
                next_token = data.get("next_page_token")
                if not next_token:
                    break
                params["page_token"] = next_token
        except requests.HTTPError as exc:
            # Some batches contain adjusted/non-standard contracts that Alpaca's
            # bars endpoint rejects. Retry symbol-by-symbol so one bad contract
            # does not fail the entire daily surface.
            response = exc.response
            if response is None or response.status_code != 400:
                raise

            for symbol in batch_symbols:
                single_params = {
                    "symbols":   symbol,
                    "timeframe": "1Day",
                    "start":     start,
                    "end":       end,
                    "limit":     10000,
                    "sort":      "asc",
                }
                try:
                    while True:
                        resp = requests.get(url, headers=HEADERS, params=single_params, timeout=30)
                        resp.raise_for_status()
                        data = resp.json()
                        for sym, bars in data.get("bars", {}).items():
                            all_bars.setdefault(sym, []).extend(bars)
                        next_token = data.get("next_page_token")
                        if not next_token:
                            break
                        single_params["page_token"] = next_token
                except requests.HTTPError as single_exc:
                    single_response = single_exc.response
                    if single_response is not None and single_response.status_code == 400:
                        print(f"Skipping invalid contract for {day}: {symbol}")
                        continue
                    raise
    return all_bars

def _parse_occ(symbol: str):
    try:
        i = next(i for i, c in enumerate(symbol) if c.isdigit())
        return (f"20{symbol[i:i+2]}-{symbol[i+2:i+4]}-{symbol[i+4:i+6]}",
                "call" if symbol[i+6] == "C" else "put",
                int(symbol[i+7:]) / 1000)
    except Exception:
        return None, None, None

def _save_csv(day: str, symbol: str, bars: dict, output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for sym, bar_list in bars.items():
        expiry, opt_type, strike = _parse_occ(sym)
        for bar in bar_list:
            rows.append({
                "date": day, "symbol": sym, "expiry": expiry,
                "type": opt_type, "strike": strike, "timestamp": bar["t"],
                "open": bar["o"], "high": bar["h"], "low": bar["l"],
                "close": bar["c"], "volume": bar["v"],
                "trade_count": bar["n"], "vwap": bar["vw"],
            })
    if not rows:
        return 0
    path = os.path.join(output_dir, f"vol_surface_{day}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)

def fetch_vol_surface(
    underlying: str,
    start_date: str,
    end_date: str,
    output_dir: str = "./data/vol_surfaces",
    max_days_to_expiry: int = 365,
):
    """
    Fetch and save the vol surface for every trading day between
    start_date and end_date (inclusive) for the given underlying.

    Contracts are fetched separately for each day so the saved daily surface is
    anchored to that day's option chain rather than one large contract universe
    built from the overall start date. Expiries are capped to a rolling horizon
    using max_days_to_expiry.
 
    Saves one CSV per day to output_dir:
        {output_dir}/{underlying}/vol_surface_{date}.csv
 
    Args:
        underlying:  Ticker symbol, e.g. "AAPL"
        start_date:  Start date string "YYYY-MM-DD"
        end_date:    End date string   "YYYY-MM-DD"
        output_dir:  Folder to save CSVs (created if it doesn't exist)
        max_days_to_expiry: Maximum days from each observation date to include
                            in the saved daily surface.
    """
    days = _trading_days(start_date, end_date)
    print(f"{underlying}: {start_date} → {end_date}  ({len(days)} trading days)")
    
    output_dir = os.path.join(output_dir, underlying)
    total_rows = 0
    for i, day in enumerate(days, 1):
        day_output_path = os.path.join(output_dir, f"vol_surface_{day}.csv")
        if os.path.exists(day_output_path):
            print(f"[{i:3}/{len(days)}] {day} cached, skipped")
            continue

        day_dt = datetime.date.fromisoformat(day)
        expiry_lte = (day_dt + datetime.timedelta(days=max_days_to_expiry)).isoformat()

        contracts = _get_contracts_for_day(underlying, day=day, expiry_lte=expiry_lte)
        if not contracts:
            print(f"[{i:3}/{len(days)}] {day} 0 contracts, 0 rows")
            continue

        bars = _get_bars_for_day(contracts, day)
        n    = _save_csv(day, underlying, bars, output_dir)
        total_rows += n
        print(f"[{i:3}/{len(days)}] {day}  {len(bars):4} contracts  {n:5} rows")
        time.sleep(60)  # avoid hitting API rate limits
 
    print(f"\nDone. {total_rows:,} rows across {len(days)} files in '{output_dir}/'")

def load_prices(tickers: list[str], start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None) -> dict[str, pd.DataFrame]:
    """
    Load historical price data for the given tickers.
    If both start and end date are null, fetch entire history available in yfinance.

    Args:
        tickers: List of ticker symbols to load data for.
        start_date: Optional start date for the historical data.
        end_date: Optional end date for the historical data.

    Returns:
        A dictionary mapping each ticker to its historical price data as a pandas DataFrame.
    """

    folder = "./data/prices"
    if not os.path.exists(folder):
        os.makedirs(folder)

    price_data = {}
    for ticker in tickers:
        tckr = yf.Ticker(ticker)
        # If no date range supplied, request the full available history
        if start_date is None and end_date is None:
            df = tckr.history(period="max")
        else:
            # yfinance accepts None for end or start; pass through the provided values
            df = tckr.history(start=start_date, end=end_date)

        price_data[ticker] = df
        df.to_csv(os.path.join(folder, f"{ticker}.csv"))

    return price_data

def load_volatility_surface(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load the volatility surface for a given ticker on future expiration dates.
    Saves the data into a data folder as a csv file.
    Only gets vol surface for current date, does not have historical vol surface data.

    Args:
        tickers: The list of ticker symbols to load the volatility surface for.
    
    Returns:
        A dictionary mapping each ticker to its volatility surface data as a pandas DataFrame.
    """

    folder = "./data/vol_surface"
    if not os.path.exists(folder):
        os.makedirs(folder)

    vol_surface_data = {}
    for ticker in tickers:
        tckr = yf.Ticker(ticker)
        options = tckr.options
        vol_data = []
        for exp in options:
            opt_chain = tckr.option_chain(exp)
            calls = opt_chain.calls
            puts = opt_chain.puts

            # add option type and expiry to the dataframes
            calls["option_type"] = "call"
            calls["expiry"] = exp
            puts["option_type"] = "put"
            puts["expiry"] = exp

            vol_data.append(calls)
            vol_data.append(puts)
        
        vol_surface_df = pd.concat(vol_data, ignore_index=True)
        vol_surface_data[ticker] = vol_surface_df
        vol_surface_df.to_csv(os.path.join(folder, f"{ticker}_vol_surface.csv"), index=False)

    return vol_surface_data

def load_3m_treasury_yield(start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None) -> pd.DataFrame:
    """
    Load historical 3 month treasury yield data.
    If both start and end date are null, fetch entire history available in yfinance.

    Args:
        start_date: Optional start date for the historical data.
        end_date: Optional end date for the historical data.

    Returns:
        A pandas DataFrame containing the historical 3 month treasury yield data.
    """

    folder = "./data/treasury_yield"
    if not os.path.exists(folder):
        os.makedirs(folder)

    tckr = yf.Ticker("^IRX") # ticker for 3 month treasury yield
    # If no date range supplied, request the full available history
    if start_date is None and end_date is None:
        df = tckr.history(period="max")
    else:
        # yfinance accepts None for end or start; pass through the provided values
        df = tckr.history(start=start_date, end=end_date)

    df.to_csv(os.path.join(folder, "3m_treasury_yield.csv"))
    return df
