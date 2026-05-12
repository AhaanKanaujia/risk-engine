import os
import datetime
from portfolio import data, position

class Portfolio:
    positions: list[position.Position]

    def __init__(self):
        self.positions = []
    
    def __repr__(self):
        return f"Portfolio(positions={self.positions})"

    def add_position(self, position: position.Position):
        self.positions.append(position)
    
    def remove_position(self, position: position.Position):
        self.positions.remove(position)
    
    def get_position_tickers_on_date(self, date: datetime.date) -> list[str]:
        stock_positions = self.get_portfolio_stock_positions_on_date(date)
        option_positions = self.get_portfolio_option_positions_on_date(date)
        stock_tickers = set(pos.ticker for pos in stock_positions)
        option_tickers = set(pos.underlying for pos in option_positions)
        return sorted(stock_tickers.union(option_tickers))

    def get_portfolio_stock_positions_on_date(self, date: datetime.date) -> list[position.Position]:
        return [pos for pos in self.positions if pos.tradable_type == "stock" and pos.date_opened <= date]

    def get_portfolio_option_positions_on_date(self, date: datetime.date) -> list[position.Position]:
        return [pos for pos in self.positions if pos.tradable_type == "option" and pos.date_opened <= date and pos.expiry >= date]

    def get_positions_on_date(self, date: datetime.date) -> list[position.Position]:
        return [pos for pos in self.positions if pos.date_opened <= date]

    def get_portfolio_creation_date(self) -> datetime.date:
        if not self.positions:
            return None
        return min(pos.date_opened for pos in self.positions)
    
    def get_portfolio_stock_quantities_on_date(self, date: datetime.date) -> dict[str, int]:
        positions = self.get_portfolio_stock_positions_on_date(date)
        quantities = {}
        for pos in positions:
            if pos.ticker not in quantities:
                quantities[pos.ticker] = 0
            quantities[pos.ticker] += pos.quantity
        return quantities

    def get_portfolio_option_quantities_on_date(self, date: datetime.date) -> dict[str, int]:
        positions = self.get_portfolio_option_positions_on_date(date)
        quantities = {}
        for pos in positions:
            if pos.ticker not in quantities:
                quantities[pos.ticker] = 0
            quantities[pos.ticker] += pos.quantity
        return quantities

    def get_position_prices_on_date(self, date: datetime.date) -> dict[str, float]:
        """
        Get the prices of the positions in the portfolio on a given date using historical price data.
        
        Args:
            date: The date on which to get the prices of the positions.
            
        Returns:
            A dictionary mapping each ticker to its price on the given date.
        """
        positions = self.get_positions_on_date(date)
        prices = {}
        for pos in positions:
            if pos.tradable_type == "stock":
                prices[pos.ticker]  = pos.get_stock_price_on_date(date)
            elif pos.tradable_type == "option":
                prices[pos.ticker] = pos.get_option_price_on_date(date)
                # also include the underlying price (needed for risk calculations)
                if pos.underlying not in prices:
                    prices[pos.underlying] = pos.get_underlying_price_on_date(date)
            else:
                raise ValueError(f"Invalid tradable type: {pos.tradable_type}")
        return prices

    def get_portfolio_unrealized_pnl_on_date(self, date: datetime.date) -> float:
        """
        Get the unrealized PnL of the portfolio on a given date using historical price data.
        
        Args:
            date: The date on which to get the unrealized PnL of the portfolio.
            
        Returns:
            The unrealized PnL of the portfolio on the given date.
        """
        positions = self.get_positions_on_date(date)
        pnl = 0.0
        for pos in positions:
            pos_pnl = pos.get_unrealized_pnl_on_date(date)
            pnl += pos_pnl
            # print(f"Unrealized PnL of position {pos.ticker} on {date} = {pos_pnl}")
        return pnl
    