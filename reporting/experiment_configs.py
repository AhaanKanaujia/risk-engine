import datetime
from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass
class ExperimentConfig:
    """
    Configuration for one reporting / backtesting experiment.
    """

    name: str
    category: str
    portfolio_builder: str
    start_date: datetime.date
    end_date: datetime.date
    window_size: int
    up_to_days: int
    alpha: float
    lambda_val: float
    num_simulations: int = 10_000
    portfolio_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        config_dict = asdict(self)
        config_dict["start_date"] = self.start_date.isoformat()
        config_dict["end_date"] = self.end_date.isoformat()

        serialized_portfolio_kwargs = {}
        for key, value in self.portfolio_kwargs.items():
            if isinstance(value, datetime.date):
                serialized_portfolio_kwargs[key] = value.isoformat()
            else:
                serialized_portfolio_kwargs[key] = value

        config_dict["portfolio_kwargs"] = serialized_portfolio_kwargs
        return config_dict

DEFAULT_EXPERIMENTS = [
    ExperimentConfig(
        name="long_equity_core_gfc_regime",
        category="equity_only",
        portfolio_builder="long_equity_core",
        start_date=datetime.date(2008, 1, 1),
        end_date=datetime.date(2009, 12, 31),
        window_size=252,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={"opened_date": datetime.date(2007, 1, 1)},
        notes="Long-only equity basket through the global financial crisis regime.",
    ),
    ExperimentConfig(
        name="long_equity_core_calm_regime",
        category="equity_only",
        portfolio_builder="long_equity_core",
        start_date=datetime.date(2016, 1, 1),
        end_date=datetime.date(2019, 12, 31),
        window_size=252,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={"opened_date": datetime.date(2015, 1, 1)},
        notes="Long-only equity portfolio in a relatively calm pre-2020 regime.",
    ),
    ExperimentConfig(
        name="long_equity_core_stress_regime",
        category="equity_only",
        portfolio_builder="long_equity_core",
        start_date=datetime.date(2020, 1, 1),
        end_date=datetime.date(2021, 12, 31),
        window_size=252,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={"opened_date": datetime.date(2015, 1, 1)},
        notes="Long-only equity portfolio through a high-volatility stress period.",
    ),
    ExperimentConfig(
        name="tech_concentrated_equity_2021_vol_regime",
        category="equity_only",
        portfolio_builder="tech_concentrated_equity",
        start_date=datetime.date(2021, 1, 1),
        end_date=datetime.date(2022, 12, 31),
        window_size=252,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={"opened_date": datetime.date(2020, 1, 1)},
        notes="Concentrated tech equity basket through the 2021-2022 high-volatility and rate-shift regime.",
    ),
    ExperimentConfig(
        name="single_name_aapl_gfc_regime",
        category="equity_only",
        portfolio_builder="single_name_equity",
        start_date=datetime.date(2008, 1, 1),
        end_date=datetime.date(2009, 12, 31),
        window_size=252,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "ticker": "AAPL",
            "quantity": 100,
            "spot_price": 120.0,
            "opened_date": datetime.date(2007, 1, 1),
        },
        notes="Single-name AAPL equity backtest through the global financial crisis regime.",
    ),
    ExperimentConfig(
        name="single_name_msft_gfc_regime",
        category="equity_only",
        portfolio_builder="single_name_equity",
        start_date=datetime.date(2008, 1, 1),
        end_date=datetime.date(2009, 12, 31),
        window_size=252,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "ticker": "MSFT",
            "quantity": 100,
            "spot_price": 30.0,
            "opened_date": datetime.date(2007, 1, 1),
        },
        notes="Single-name MSFT equity backtest through the global financial crisis regime.",
    ),
    ExperimentConfig(
        name="single_name_aapl_2021_vol_regime",
        category="equity_only",
        portfolio_builder="single_name_equity",
        start_date=datetime.date(2021, 1, 1),
        end_date=datetime.date(2022, 12, 31),
        window_size=252,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "ticker": "AAPL",
            "quantity": 100,
            "spot_price": 130.0,
            "opened_date": datetime.date(2020, 1, 1),
        },
        notes="Single-name AAPL equity backtest through the 2021-2022 high-volatility regime.",
    ),
    ExperimentConfig(
        name="single_name_msft_2021_vol_regime",
        category="equity_only",
        portfolio_builder="single_name_equity",
        start_date=datetime.date(2021, 1, 1),
        end_date=datetime.date(2022, 12, 31),
        window_size=252,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "ticker": "MSFT",
            "quantity": 100,
            "spot_price": 220.0,
            "opened_date": datetime.date(2020, 1, 1),
        },
        notes="Single-name MSFT equity backtest through the 2021-2022 high-volatility regime.",
    ),
    ExperimentConfig(
        name="covered_call_aapl_local_regime",
        category="equity_option_combinations",
        portfolio_builder="covered_call_aapl",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 2, 20),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Covered-call portfolio combining long AAPL stock with a short AAPL call in the early local stress regime.",
    ),
    ExperimentConfig(
        name="protective_put_aapl_local_stress_regime",
        category="equity_option_combinations",
        portfolio_builder="protective_put_aapl",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 2, 20),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Protective-put portfolio on AAPL in the early local stress regime.",
    ),
    ExperimentConfig(
        name="collar_aapl_local_stress_regime",
        category="equity_option_combinations",
        portfolio_builder="collar_aapl",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 2, 20),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="AAPL collar portfolio in the early local stress regime.",
    ),
    ExperimentConfig(
        name="delta_neutral_conversion_aapl_local_stress_regime",
        category="equity_option_combinations",
        portfolio_builder="delta_neutral_conversion_aapl",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 2, 20),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Approximately delta- and gamma-neutral conversion-style portfolio on AAPL in the early local stress regime.",
    ),
    ExperimentConfig(
        name="reverse_conversion_msft_local_transition_regime",
        category="equity_option_combinations",
        portfolio_builder="reverse_conversion_msft",
        start_date=datetime.date(2026, 2, 23),
        end_date=datetime.date(2026, 4, 21),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 2, 23),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Approximately delta- and gamma-neutral reverse-conversion-style portfolio on MSFT in a later local transition regime.",
    ),
    ExperimentConfig(
        name="tech_collar_basket_local_transition_regime",
        category="equity_option_combinations",
        portfolio_builder="tech_collar_basket",
        start_date=datetime.date(2026, 2, 23),
        end_date=datetime.date(2026, 4, 21),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 2, 23),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Hedged technology basket with collar overlays in a later local transition regime.",
    ),
    ExperimentConfig(
        name="long_gamma_tech_stress_regime",
        category="options_only",
        portfolio_builder="long_gamma_tech",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 4, 30),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Long-gamma option portfolio over the local options-data regime.",
    ),
    ExperimentConfig(
        name="short_vol_pair_stress_regime",
        category="options_only",
        portfolio_builder="short_vol_pair",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 4, 30),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Short-volatility option portfolio over the local options-data regime.",
    ),
    ExperimentConfig(
        name="long_straddle_aapl_stress_regime",
        category="options_only",
        portfolio_builder="long_straddle_aapl",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 4, 30),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Long AAPL straddle in the local stress regime to test pure long-vol convexity.",
    ),
    ExperimentConfig(
        name="short_strangle_tech_stress_regime",
        category="options_only",
        portfolio_builder="short_strangle_tech",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 4, 30),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Short-strangle technology portfolio in the local stress regime to test dispersed short-vol exposure.",
    ),
    ExperimentConfig(
        name="protective_put_basket_stress_regime",
        category="options_only",
        portfolio_builder="protective_put_basket",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 4, 30),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Long-put downside hedge basket in the local stress regime.",
    ),
    ExperimentConfig(
        name="bear_put_spread_aapl_stress_regime",
        category="options_only",
        portfolio_builder="bear_put_spread_aapl",
        start_date=datetime.date(2026, 1, 2),
        end_date=datetime.date(2026, 4, 30),
        window_size=100,
        up_to_days=10,
        alpha=0.95,
        lambda_val=0.99,
        num_simulations=10_000,
        portfolio_kwargs={
            "opened_date": datetime.date(2026, 1, 2),
            "expiry": datetime.date(2026, 6, 27),
        },
        notes="Directional bearish put-spread portfolio on AAPL in the local stress regime.",
    ),
]
