import datetime

from typing import Optional
from pricing import black_scholes
from risk import utils, cache

class Position:
    ticker: str
    quantity: int # positive for long, negative for short
    tradable_type: str # stock or option
    spot_price: float # price at which the position was opened
    date_opened: datetime.date # date when the position was opened

    # option specific fields
    option_type: Optional[str] = None # call or put
    strike: Optional[float] = None
    expiry: Optional[datetime.date] = None
    underlying: Optional[str] = None

    def __init__(self, ticker, quantity, tradable_type, spot_price, date_opened, option_type=None, strike=None, expiry=None, underlying=None):
        self.ticker = ticker
        self.quantity = quantity
        self.tradable_type = tradable_type
        self.spot_price = spot_price
        self.date_opened = date_opened
        self.option_type = option_type
        self.strike = strike
        self.expiry = expiry
        self.underlying = underlying
    
    def __repr__(self):
        return f"""
            Position(
                ticker={self.ticker},
                quantity={self.quantity},
                tradable_type={self.tradable_type},
                spot_price={self.spot_price},
                date_opened={self.date_opened},
                option_type={self.option_type},
                strike={self.strike},
                expiry={self.expiry},
                underlying={self.underlying}
            )
        """

    def get_underlying_price_on_date(self, date: datetime.date) -> float:
        """
        Get the price of the underlying stock for an option position on a given date using historical price data.
        Read the price data from the data folder where it is stored as a csv file.

        Args:
            date: The date on which to get the price of the underlying stock for the option position.

        Returns:
            The price of the underlying stock for the option position on the given date.
        """

        if self.underlying is None:
            raise ValueError(f"Option position must have underlying field set.")
        
        df = cache.get_price_history(self.underlying)
        price_data = df.loc[df.index <= date].tail(1)

        if price_data.empty:
            raise ValueError(f"No price data available for underlying ticker {self.underlying} on or before date {date}.")
        
        return price_data.iloc[0]["Close"]
    
    def get_stock_price_on_date(self, date: datetime.date) -> float:
        """
        Get stock position data on a given date using historical price data.
        Read the price data from the data folder where it is stored as a csv file.

        Args:
            date: The date on which to get the price of the position.
        
        Returns:
            The price of the position on the given date.
        """

        df = cache.get_price_history(self.ticker)
        price_data = df.loc[df.index <= date].tail(1)

        if price_data.empty:
            raise ValueError(f"No price data available for ticker {self.ticker} on or before date {date}.")
        
        return price_data.iloc[0]["Close"]

    def get_option_implied_vol_on_date(self, date: datetime.date) -> float:
        """
        Get the implied volatility of the option position on a given date using historical price data and implied volatility surface data.
        Read the price data and implied volatility surface data from the data folder where it is stored as a csv file.

        Args:
            date: The date on which to get the implied volatility of the option position.
        
        Returns:
            The implied volatility of the option position on the given date.
        """

        if self.option_type is None or self.strike is None or self.expiry is None or self.underlying is None:
            raise ValueError(f"Option position must have option_type, strike, expiry, and underlying fields set.")
        
        implied_vol = black_scholes.option_implied_vol(
            date=date,
            ticker=self.underlying,
            expiry=self.expiry,
            option_type=self.option_type,
            strike=self.strike
        )

        return implied_vol
    
    def get_option_price_on_date(self, date: datetime.date) -> float:
        """
        Get option position data on a given date using historical price data and implied volatility surface data.
        Read the price data and implied volatility surface data from the data folder where it is stored as a csv file.

        Args:
            date: The date on which to get the price of the option position.
        
        Returns:
            The price of the option position on the given date.
        """

        if self.option_type is None or self.strike is None or self.expiry is None or self.underlying is None:
            raise ValueError(f"Option position must have option_type, strike, expiry, and underlying fields set.")
        
        underlying_price_data = cache.get_price_history(self.underlying)
        underlying_price_on_date = underlying_price_data.loc[underlying_price_data.index <= date].tail(1)

        if underlying_price_on_date.empty:
            raise ValueError(f"No price data available for underlying ticker {self.underlying} on or before date {date}.")
    
        underlying_price_on_date = underlying_price_on_date.iloc[0]["Close"]

        implied_vol = black_scholes.option_implied_vol(
            date=date,
            ticker=self.underlying,
            expiry=self.expiry,
            option_type=self.option_type,
            strike=self.strike
        )

        option_price_on_date = black_scholes.option_price(
            S=underlying_price_on_date,
            K=self.strike,
            T=(self.expiry - date).days / 252, # convert to years assuming 252 trading days in a year
            r=utils.get_risk_free_rate(date),
            sigma=implied_vol,
            option_type=self.option_type
        )

        return option_price_on_date
    
    def get_unrealized_pnl_on_date(self, date: datetime.date) -> float:
        """
        Get the unrealized PnL of the position on a given date using historical price data and implied volatility surface data.
        Read the price data and implied volatility surface data from the data folder where it is stored as a csv file.

        Args:
            date: The date on which to get the unrealized PnL of the position.
        
        Returns:
            The unrealized PnL of the position on the given date.
        """

        if self.tradable_type == "stock":
            price_on_date = self.get_stock_price_on_date(date)
        elif self.tradable_type == "option":
            price_on_date = self.get_option_price_on_date(date)
        else:
            raise ValueError(f"Invalid tradable type: {self.tradable_type}")
        
        return (price_on_date - self.spot_price) * self.quantity
    
