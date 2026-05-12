import datetime
import csv

from portfolio import portfolio
from portfolio import position

def load_portfolio_from_input_file(filepath: str) -> portfolio.Portfolio:
    """
    Load a portfolio from a given input file. The input file should be a csv with the following columns:
    ticker, quantity, tradable_type, spot_price, date_opened, option_type, strike, expiry, underlying

    Args:
        filepath: The path to the input csv file containing the portfolio positions.
    
    Returns:
        A Portfolio object containing the positions specified in the input file.
    """

    pf = portfolio.Portfolio()

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = position.Position(
                ticker=row["ticker"],
                quantity=int(row["quantity"]),
                tradable_type=row["tradable_type"],
                spot_price=float(row["spot_price"]),
                date_opened=datetime.datetime.strptime(row["date_opened"], "%Y-%m-%d").date(),
                option_type=row.get("option_type"),
                strike=float(row["strike"]) if row.get("strike") else None,
                expiry=datetime.datetime.strptime(row["expiry"], "%Y-%m-%d").date() if row.get("expiry") else None,
                underlying=row.get("underlying")
            )
            pf.add_position(pos)
    
    return pf