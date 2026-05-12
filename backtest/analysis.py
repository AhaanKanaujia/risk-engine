import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom

from backtest import test

def summarize_backtest_results(
    results_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
    output_path: str = "./results/engine/backtest_summary.csv",
) -> pd.DataFrame:
    """
    Summarize backtest exceedance results across all methods and both risk metrics.

    Args:
        results_df: A dataframe containing alpha plus row-level exceedance flags created by
            the backtest module.
        p_value_threshold: The significance threshold used to accept or reject the Kupiec POF test.

    Returns:
        A dataframe with one row per method-risk-metric combination containing observation
        counts, expected and actual exceedance counts/percentages, and Kupiec POF test results.
    """

    if results_df.empty:
        return pd.DataFrame()

    if "alpha" not in results_df.columns:
        raise ValueError("results_df must contain an alpha column.")

    alpha_values = results_df["alpha"].dropna().unique()
    if len(alpha_values) != 1:
        raise ValueError("results_df must contain exactly one alpha value for analysis.")

    alpha = float(alpha_values[0])

    exceedance_columns = [
        col for col in results_df.columns
        if col.endswith("_var_exceeded") or col.endswith("_es_exceeded")
    ]

    summary_rows = []
    for exceedance_col in sorted(exceedance_columns):
        base_label = exceedance_col.removesuffix("_exceeded")
        if base_label.endswith("_var"):
            method = base_label.removesuffix("_var")
            risk_metric = "var"
        elif base_label.endswith("_es"):
            method = base_label.removesuffix("_es")
            risk_metric = "es"
        else:
            continue

        valid_observations = results_df[exceedance_col].dropna()
        num_observations = int(len(valid_observations))
        actual_exceedances = int(valid_observations.sum())

        expected_exceedance_pct = 1 - alpha
        expected_exceedances = int(round(num_observations * expected_exceedance_pct))
        actual_exceedance_pct = actual_exceedances / num_observations if num_observations > 0 else 0.0

        kupiec_lr_stat, kupiec_p_value = test.kupiec_proportion_of_failures_test(
            num_observations=num_observations,
            num_exceedances=actual_exceedances,
            alpha=alpha,
        )
        kupiec_result = "rejected" if pd.notna(kupiec_p_value) and kupiec_p_value < p_value_threshold else "accepted"
        exact_binomial_p_value, exact_ci_lower, exact_ci_upper = test.exact_binomial_coverage_test(
            num_observations=num_observations,
            num_exceedances=actual_exceedances,
            alpha=alpha,
        )
        exact_binomial_result = (
            "rejected"
            if pd.notna(exact_binomial_p_value) and exact_binomial_p_value < p_value_threshold
            else "accepted"
        )

        if risk_metric == "var":
            traffic_light, green_upper, yellow_upper = test.basel_traffic_light_classification(
                num_observations=num_observations,
                num_exceedances=actual_exceedances,
                alpha=alpha,
            )
        else:
            traffic_light, green_upper, yellow_upper = "n/a", np.nan, np.nan

        summary_rows.append(
            {
                "method": method,
                "risk_metric": risk_metric,
                "num_observations": num_observations,
                "expected_exceedances": expected_exceedances,
                "expected_exceedance_pct": expected_exceedance_pct,
                "actual_exceedances": actual_exceedances,
                "actual_exceedance_pct": actual_exceedance_pct,
                "kupiec_lr_stat": kupiec_lr_stat,
                "kupiec_p_value": kupiec_p_value,
                "kupiec_result": kupiec_result,
                "exact_binomial_p_value": exact_binomial_p_value,
                "exact_binomial_ci_lower": exact_ci_lower,
                "exact_binomial_ci_upper": exact_ci_upper,
                "exact_binomial_result": exact_binomial_result,
                "traffic_light": traffic_light,
                "traffic_light_green_upper": green_upper,
                "traffic_light_yellow_upper": yellow_upper,
            }
        )
    
    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    return summary_df

def _get_method_metric_pairs(results_df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Infer method/risk-metric pairs from exceedance columns in the backtest results dataframe.

    Args:
        results_df: The row-level backtest results dataframe.

    Returns:
        A sorted list of (method, risk_metric) pairs.
    """

    exceedance_columns = [
        col for col in results_df.columns
        if col.endswith("_var_exceeded") or col.endswith("_es_exceeded")
    ]

    pairs = []
    for exceedance_col in sorted(exceedance_columns):
        base_label = exceedance_col.removesuffix("_exceeded")
        if base_label.endswith("_var"):
            pairs.append((base_label.removesuffix("_var"), "var"))
        elif base_label.endswith("_es"):
            pairs.append((base_label.removesuffix("_es"), "es"))

    return pairs

def _prepare_backtest_plot_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a sorted plotting dataframe with a datetime index and realized loss column.

    Args:
        results_df: The row-level backtest results dataframe.

    Returns:
        A dataframe indexed by datetime date and augmented with realized_loss.
    """

    plot_df = results_df.sort_values("date").copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df["realized_loss"] = -plot_df["realized_pnl"]
    return plot_df.set_index("date")

def plot_backtest_exceedance_windows(
    results_df: pd.DataFrame,
    output_path: str = "./results/engine/backtest_exceedance_windows.png",
) -> str:
    """
    Plot rolling exceedance counts over a trailing 252-observation window for all methods and both risk metrics.

    Args:
        results_df: The row-level backtest results dataframe with exceedance flags.
        output_path: The file path where the figure should be saved.

    Returns:
        The saved file path.
    """

    if results_df.empty:
        raise ValueError("results_df must not be empty.")

    required_columns = {"date", "alpha"}
    missing_columns = required_columns - set(results_df.columns)
    if missing_columns:
        raise ValueError(
            f"results_df must contain the following columns: {sorted(required_columns)}. "
            f"Missing columns: {sorted(missing_columns)}."
        )

    alpha_values = results_df["alpha"].dropna().unique()
    if len(alpha_values) != 1:
        raise ValueError("results_df must contain exactly one alpha value for plotting.")

    alpha = float(alpha_values[0])
    method_metric_pairs = _get_method_metric_pairs(results_df)

    if len(method_metric_pairs) != 8:
        raise ValueError("Expected exactly 8 method/risk-metric combinations for the rolling 1-year exceedance plot.")

    plot_df = _prepare_backtest_plot_df(results_df)

    rolling_window = 252
    expected_exceedances = rolling_window * (1 - alpha)
    upper_bound = float(binom.ppf(0.95, rolling_window, 1 - alpha))

    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)
    fig.suptitle("Trailing 252-Observation Exceedance Counts", fontsize=18)

    for ax, (method, risk_metric) in zip(axes.flatten(), method_metric_pairs):
        exceedance_col = f"{method}_{risk_metric}_exceeded"
        rolling_exceedances = plot_df[exceedance_col].rolling(window=rolling_window, min_periods=rolling_window).sum()

        ax.plot(plot_df.index, rolling_exceedances, color="steelblue", linewidth=1.8, label="Actual")
        ax.axhline(expected_exceedances, color="red", linestyle="--", linewidth=1.8, label=f"Expected ({expected_exceedances:.1f})")
        ax.axhline(upper_bound, color="orange", linestyle=":", linewidth=1.8, label=f"95% upper bound ({upper_bound:.0f})")

        ax.set_title(f"{method} - {risk_metric.upper()}")
        ax.set_ylabel("Exceedances in trailing 252-observation window")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("Date")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path

def plot_backtest_exceedance_scatter(
    results_df: pd.DataFrame,
    output_path: str = "./results/engine/backtest_exceedance_scatter.png",
) -> str:
    """
    Plot realized loss versus forecast VaR/ES estimate for exceeded points across all methods and metrics.

    Args:
        results_df: The row-level backtest results dataframe with realized pnl and exceedance flags.
        output_path: The file path where the figure should be saved.

    Returns:
        The saved file path.
    """

    if results_df.empty:
        raise ValueError("results_df must not be empty.")

    if "realized_pnl" not in results_df.columns:
        raise ValueError("results_df must contain a realized_pnl column.")

    method_metric_pairs = _get_method_metric_pairs(results_df)
    if len(method_metric_pairs) != 8:
        raise ValueError("Expected exactly 8 method/risk-metric combinations for the exceedance scatter plot.")

    plot_df = _prepare_backtest_plot_df(results_df)

    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    fig.suptitle("Exceeded Forecasts: Realized Loss vs Forecast Threshold", fontsize=18)

    for ax, (method, risk_metric) in zip(axes.flatten(), method_metric_pairs):
        threshold_col = f"{method}_{risk_metric}"
        exceeded_col = f"{threshold_col}_exceeded"
        exceeded_points = plot_df[plot_df[exceeded_col] == 1]

        ax.set_title(f"{method} - {risk_metric.upper()}")
        ax.set_xlabel(f"{risk_metric.upper()} estimate ($)")
        ax.set_ylabel("Realized loss ($)")
        ax.grid(True, alpha=0.3)

        if exceeded_points.empty:
            ax.text(0.5, 0.5, "No exceedances", ha="center", va="center", transform=ax.transAxes)
            continue

        x = exceeded_points[threshold_col].to_numpy(dtype=float)
        y = exceeded_points["realized_loss"].to_numpy(dtype=float)

        ax.scatter(x, y, color="crimson", alpha=0.55, s=28, label=f"{len(exceeded_points)} exceedances")

        lower_bound = min(np.nanmin(x), np.nanmin(y))
        upper_bound = max(np.nanmax(x), np.nanmax(y))
        ax.plot(
            [lower_bound, upper_bound],
            [lower_bound, upper_bound],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Loss = {risk_metric.upper()}",
        )
        ax.legend(loc="lower right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path

def plot_backtest_hit_sequence(
    results_df: pd.DataFrame,
    output_path: str = "./results/engine/backtest_hit_sequence.png",
) -> str:
    """
    Plot the 0/1 exceedance hit sequence over time for all methods and both risk metrics.

    Args:
        results_df: The row-level backtest results dataframe with exceedance flags.
        output_path: The file path where the figure should be saved.

    Returns:
        The saved file path.
    """

    if results_df.empty:
        raise ValueError("results_df must not be empty.")

    method_metric_pairs = _get_method_metric_pairs(results_df)
    if len(method_metric_pairs) != 8:
        raise ValueError("Expected exactly 8 method/risk-metric combinations for the hit-sequence plot.")

    plot_df = _prepare_backtest_plot_df(results_df)

    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True, sharey=True)
    fig.suptitle("Exceedance Hit Sequence", fontsize=18)

    for ax, (method, risk_metric) in zip(axes.flatten(), method_metric_pairs):
        exceeded_col = f"{method}_{risk_metric}_exceeded"
        hits = plot_df[exceeded_col]

        ax.step(plot_df.index, hits, where="mid", color="steelblue", linewidth=1.2)
        breach_dates = plot_df.index[hits == 1]
        if len(breach_dates) > 0:
            ax.scatter(breach_dates, np.ones(len(breach_dates)), color="crimson", s=14, zorder=3)

        ax.set_title(f"{method} - {risk_metric.upper()}")
        ax.set_ylabel("Hit")
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Date")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path

def plot_backtest_realized_loss_vs_threshold(
    results_df: pd.DataFrame,
    output_path: str = "./results/engine/backtest_realized_loss_vs_threshold.png",
) -> str:
    """
    Plot realized loss and forecast threshold over time for all methods and both risk metrics.

    Args:
        results_df: The row-level backtest results dataframe with realized pnl and forecast thresholds.
        output_path: The file path where the figure should be saved.

    Returns:
        The saved file path.
    """

    if results_df.empty:
        raise ValueError("results_df must not be empty.")

    if "realized_pnl" not in results_df.columns:
        raise ValueError("results_df must contain a realized_pnl column.")

    method_metric_pairs = _get_method_metric_pairs(results_df)
    if len(method_metric_pairs) != 8:
        raise ValueError("Expected exactly 8 method/risk-metric combinations for the loss-vs-threshold plot.")

    plot_df = _prepare_backtest_plot_df(results_df)

    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)
    fig.suptitle("Realized Loss vs Forecast Threshold Over Time", fontsize=18)

    for ax, (method, risk_metric) in zip(axes.flatten(), method_metric_pairs):
        threshold_col = f"{method}_{risk_metric}"
        exceeded_col = f"{threshold_col}_exceeded"

        ax.plot(plot_df.index, plot_df["realized_loss"], color="steelblue", linewidth=1.3, label="Realized loss")
        ax.plot(plot_df.index, plot_df[threshold_col], color="black", linestyle="--", linewidth=1.4, label=risk_metric.upper())

        exceeded_points = plot_df[plot_df[exceeded_col] == 1]
        if not exceeded_points.empty:
            ax.scatter(
                exceeded_points.index,
                exceeded_points["realized_loss"],
                color="crimson",
                s=16,
                zorder=3,
                label="Exceedance",
            )

        ax.set_title(f"{method} - {risk_metric.upper()}")
        ax.set_ylabel("Loss / threshold ($)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("Date")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path
