import datetime

from portfolio import portfolio, position

def build_long_equity_core(
    opened_date: datetime.date = datetime.date(2015, 1, 1),
) -> portfolio.Portfolio:
    """
    Build a long-only core equity portfolio.

    Args:
        opened_date: The date on which all positions are assumed to be opened.

    Returns:
        A portfolio containing long equity positions.
    """

    pf = portfolio.Portfolio()
    pf.add_position(position.Position(ticker="AAPL", quantity=100, tradable_type="stock", spot_price=250.0, date_opened=opened_date))
    pf.add_position(position.Position(ticker="MSFT", quantity=50, tradable_type="stock", spot_price=475.0, date_opened=opened_date))
    pf.add_position(position.Position(ticker="GOOGL", quantity=30, tradable_type="stock", spot_price=300.0, date_opened=opened_date))
    return pf

def build_single_name_equity(
    ticker: str,
    quantity: int,
    spot_price: float,
    opened_date: datetime.date = datetime.date(2015, 1, 1),
) -> portfolio.Portfolio:
    """
    Build a single-name equity portfolio.

    Args:
        ticker: The ticker to hold.
        quantity: The signed stock quantity.
        spot_price: The assumed opening spot price.
        opened_date: The date on which the position is assumed to be opened.

    Returns:
        A portfolio containing one equity position.
    """

    pf = portfolio.Portfolio()
    pf.add_position(
        position.Position(
            ticker=ticker,
            quantity=quantity,
            tradable_type="stock",
            spot_price=spot_price,
            date_opened=opened_date,
        )
    )
    return pf

def build_tech_concentrated_equity(
    opened_date: datetime.date = datetime.date(2015, 1, 1),
) -> portfolio.Portfolio:
    """
    Build a more concentrated technology equity portfolio.

    Args:
        opened_date: The date on which all positions are assumed to be opened.

    Returns:
        A portfolio containing concentrated long technology equity positions.
    """

    pf = portfolio.Portfolio()
    pf.add_position(position.Position(ticker="AAPL", quantity=150, tradable_type="stock", spot_price=250.0, date_opened=opened_date))
    pf.add_position(position.Position(ticker="MSFT", quantity=100, tradable_type="stock", spot_price=475.0, date_opened=opened_date))
    pf.add_position(position.Position(ticker="GOOGL", quantity=60, tradable_type="stock", spot_price=300.0, date_opened=opened_date))
    return pf

def build_covered_call_aapl(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build a covered-call portfolio on AAPL.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date of the short call option.

    Returns:
        A portfolio containing long AAPL stock and a short AAPL call.
    """

    pf = portfolio.Portfolio()
    pf.add_position(position.Position(ticker="AAPL", quantity=100, tradable_type="stock", spot_price=250.0, date_opened=opened_date))
    pf.add_position(
        position.Position(
            ticker="AAPL_C_250_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=40.0,
            option_type="call",
            strike=250.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    return pf


def build_protective_put_aapl(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build a protective-put portfolio on AAPL.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date of the long put option.

    Returns:
        A portfolio containing long AAPL stock and a long downside put.
    """

    pf = portfolio.Portfolio()
    pf.add_position(position.Position(ticker="AAPL", quantity=100, tradable_type="stock", spot_price=250.0, date_opened=opened_date))
    pf.add_position(
        position.Position(
            ticker="AAPL_P_235_HEDGE",
            quantity=1,
            tradable_type="option",
            spot_price=22.0,
            option_type="put",
            strike=235.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    return pf


def build_collar_aapl(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build a collar portfolio on AAPL.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the hedge options.

    Returns:
        A portfolio containing long AAPL stock, a long put, and a short call.
    """

    pf = portfolio.Portfolio()
    pf.add_position(position.Position(ticker="AAPL", quantity=100, tradable_type="stock", spot_price=250.0, date_opened=opened_date))
    pf.add_position(
        position.Position(
            ticker="AAPL_P_235_COLLAR_LONG",
            quantity=1,
            tradable_type="option",
            spot_price=22.0,
            option_type="put",
            strike=235.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="AAPL_C_270_COLLAR_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=28.0,
            option_type="call",
            strike=270.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    return pf


def build_delta_neutral_conversion_aapl(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build an approximately delta- and gamma-neutral conversion-style portfolio.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the call and put.

    Returns:
        A portfolio containing long stock, long put, and short call at a common strike.
    """

    pf = portfolio.Portfolio()
    pf.add_position(position.Position(ticker="AAPL", quantity=100, tradable_type="stock", spot_price=250.0, date_opened=opened_date))
    pf.add_position(
        position.Position(
            ticker="AAPL_P_250_CONVERSION_LONG",
            quantity=1,
            tradable_type="option",
            spot_price=35.0,
            option_type="put",
            strike=250.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="AAPL_C_250_CONVERSION_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=40.0,
            option_type="call",
            strike=250.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    return pf


def build_reverse_conversion_msft(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build an approximately delta- and gamma-neutral reverse-conversion-style portfolio.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the call and put.

    Returns:
        A portfolio containing short stock, long call, and short put at a common strike.
    """

    pf = portfolio.Portfolio()
    pf.add_position(position.Position(ticker="MSFT", quantity=-100, tradable_type="stock", spot_price=475.0, date_opened=opened_date))
    pf.add_position(
        position.Position(
            ticker="MSFT_C_475_REVCON_LONG",
            quantity=1,
            tradable_type="option",
            spot_price=30.0,
            option_type="call",
            strike=475.0,
            expiry=expiry,
            underlying="MSFT",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="MSFT_P_475_REVCON_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=28.0,
            option_type="put",
            strike=475.0,
            expiry=expiry,
            underlying="MSFT",
            date_opened=opened_date,
        )
    )
    return pf


def build_tech_collar_basket(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build a hedged technology basket using stock plus collar overlays.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the hedge options.

    Returns:
        A portfolio containing long stock plus collar overlays on AAPL and MSFT.
    """

    pf = portfolio.Portfolio()
    pf.add_position(position.Position(ticker="AAPL", quantity=100, tradable_type="stock", spot_price=250.0, date_opened=opened_date))
    pf.add_position(position.Position(ticker="MSFT", quantity=75, tradable_type="stock", spot_price=475.0, date_opened=opened_date))
    pf.add_position(
        position.Position(
            ticker="AAPL_P_235_BASKET_LONG",
            quantity=1,
            tradable_type="option",
            spot_price=22.0,
            option_type="put",
            strike=235.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="AAPL_C_270_BASKET_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=28.0,
            option_type="call",
            strike=270.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="MSFT_P_450_BASKET_LONG",
            quantity=1,
            tradable_type="option",
            spot_price=24.0,
            option_type="put",
            strike=450.0,
            expiry=expiry,
            underlying="MSFT",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="MSFT_C_510_BASKET_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=26.0,
            option_type="call",
            strike=510.0,
            expiry=expiry,
            underlying="MSFT",
            date_opened=opened_date,
        )
    )
    return pf

def build_long_gamma_tech(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build a long-gamma technology portfolio using long call and put positions.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the options.

    Returns:
        A portfolio containing long option positions on AAPL and MSFT.
    """

    pf = portfolio.Portfolio()
    pf.add_position(
        position.Position(
            ticker="AAPL_C_250_LONG",
            quantity=2,
            tradable_type="option",
            spot_price=40.0,
            option_type="call",
            strike=250.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="MSFT_P_475_LONG",
            quantity=2,
            tradable_type="option",
            spot_price=30.0,
            option_type="put",
            strike=475.0,
            expiry=expiry,
            underlying="MSFT",
            date_opened=opened_date,
        )
    )
    return pf

def build_short_vol_pair(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build a short-volatility portfolio using short option positions.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the options.

    Returns:
        A portfolio containing short option positions on AAPL and MSFT.
    """

    pf = portfolio.Portfolio()
    pf.add_position(
        position.Position(
            ticker="AAPL_C_250_SHORT",
            quantity=-2,
            tradable_type="option",
            spot_price=40.0,
            option_type="call",
            strike=250.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="MSFT_P_475_SHORT",
            quantity=-2,
            tradable_type="option",
            spot_price=30.0,
            option_type="put",
            strike=475.0,
            expiry=expiry,
            underlying="MSFT",
            date_opened=opened_date,
        )
    )
    return pf


def build_long_straddle_aapl(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build a long-straddle portfolio on AAPL.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the options.

    Returns:
        A portfolio containing a long ATM call and long ATM put on AAPL.
    """

    pf = portfolio.Portfolio()
    pf.add_position(
        position.Position(
            ticker="AAPL_C_250_STRADDLE_LONG",
            quantity=1,
            tradable_type="option",
            spot_price=40.0,
            option_type="call",
            strike=250.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="AAPL_P_250_STRADDLE_LONG",
            quantity=1,
            tradable_type="option",
            spot_price=35.0,
            option_type="put",
            strike=250.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    return pf


def build_short_strangle_tech(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build a short-strangle portfolio across large-cap technology names.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the options.

    Returns:
        A portfolio containing short OTM call/put positions.
    """

    pf = portfolio.Portfolio()
    pf.add_position(
        position.Position(
            ticker="AAPL_C_270_STRANGLE_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=28.0,
            option_type="call",
            strike=270.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="AAPL_P_230_STRANGLE_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=24.0,
            option_type="put",
            strike=230.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="MSFT_C_510_STRANGLE_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=26.0,
            option_type="call",
            strike=510.0,
            expiry=expiry,
            underlying="MSFT",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="MSFT_P_440_STRANGLE_SHORT",
            quantity=-1,
            tradable_type="option",
            spot_price=22.0,
            option_type="put",
            strike=440.0,
            expiry=expiry,
            underlying="MSFT",
            date_opened=opened_date,
        )
    )
    return pf


def build_protective_put_basket(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build an options-only downside-hedge style portfolio using long puts.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the options.

    Returns:
        A portfolio containing long downside put exposure on AAPL and MSFT.
    """

    pf = portfolio.Portfolio()
    pf.add_position(
        position.Position(
            ticker="AAPL_P_235_HEDGE_LONG",
            quantity=2,
            tradable_type="option",
            spot_price=22.0,
            option_type="put",
            strike=235.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="MSFT_P_450_HEDGE_LONG",
            quantity=2,
            tradable_type="option",
            spot_price=24.0,
            option_type="put",
            strike=450.0,
            expiry=expiry,
            underlying="MSFT",
            date_opened=opened_date,
        )
    )
    return pf


def build_bear_put_spread_aapl(
    opened_date: datetime.date = datetime.date(2026, 1, 2),
    expiry: datetime.date = datetime.date(2026, 6, 27),
) -> portfolio.Portfolio:
    """
    Build a directional bearish put-spread portfolio on AAPL.

    Args:
        opened_date: The date on which all positions are assumed to be opened.
        expiry: The expiry date shared by the options.

    Returns:
        A portfolio containing a long put and offsetting short lower-strike put.
    """

    pf = portfolio.Portfolio()
    pf.add_position(
        position.Position(
            ticker="AAPL_P_250_SPREAD_LONG",
            quantity=2,
            tradable_type="option",
            spot_price=35.0,
            option_type="put",
            strike=250.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    pf.add_position(
        position.Position(
            ticker="AAPL_P_220_SPREAD_SHORT",
            quantity=-2,
            tradable_type="option",
            spot_price=18.0,
            option_type="put",
            strike=220.0,
            expiry=expiry,
            underlying="AAPL",
            date_opened=opened_date,
        )
    )
    return pf

PORTFOLIO_BUILDERS = {
    "long_equity_core": build_long_equity_core,
    "single_name_equity": build_single_name_equity,
    "tech_concentrated_equity": build_tech_concentrated_equity,
    "covered_call_aapl": build_covered_call_aapl,
    "protective_put_aapl": build_protective_put_aapl,
    "collar_aapl": build_collar_aapl,
    "delta_neutral_conversion_aapl": build_delta_neutral_conversion_aapl,
    "reverse_conversion_msft": build_reverse_conversion_msft,
    "tech_collar_basket": build_tech_collar_basket,
    "long_gamma_tech": build_long_gamma_tech,
    "short_vol_pair": build_short_vol_pair,
    "long_straddle_aapl": build_long_straddle_aapl,
    "short_strangle_tech": build_short_strangle_tech,
    "protective_put_basket": build_protective_put_basket,
    "bear_put_spread_aapl": build_bear_put_spread_aapl,
}
