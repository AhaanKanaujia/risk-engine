import json
import os

import pandas as pd

from backtest import analysis, backtest, forecast
from reporting import experiment_configs, portfolio_library

def _get_results_dir(config: experiment_configs.ExperimentConfig, base_results_dir: str) -> str:
    """
    Build the output directory for an experiment run.

    Args:
        config: The experiment configuration.
        base_results_dir: The root results directory.

    Returns:
        The experiment-specific results directory path.
    """

    return os.path.join(base_results_dir, config.category, config.name)


def _get_expected_artifact_paths(results_dir: str) -> dict[str, str]:
    return {
        "metadata_path": os.path.join(results_dir, "metadata.json"),
        "forecast_path": os.path.join(results_dir, "forecast_results.csv"),
        "realized_path": os.path.join(results_dir, "realized_results.csv"),
        "backtested_path": os.path.join(results_dir, "backtested_results.csv"),
        "summary_path": os.path.join(results_dir, "summary.csv"),
        "exceedance_windows_path": os.path.join(results_dir, "exceedance_windows.png"),
        "exceedance_scatter_path": os.path.join(results_dir, "exceedance_scatter.png"),
        "hit_sequence_path": os.path.join(results_dir, "hit_sequence.png"),
        "realized_loss_vs_threshold_path": os.path.join(results_dir, "realized_loss_vs_threshold.png"),
    }


def _artifacts_exist(artifact_paths: dict[str, str]) -> bool:
    return all(os.path.exists(path) for path in artifact_paths.values())

def run_experiment(
    config: experiment_configs.ExperimentConfig,
    base_results_dir: str = "./results",
    force_rerun: bool = False,
) -> dict[str, object]:
    """
    Run one reporting experiment end-to-end and save all artifacts.

    Args:
        config: The experiment configuration.
        base_results_dir: The directory under which experiment outputs are saved.
        force_rerun: Whether to recompute the scenario even if saved outputs exist.

    Returns:
        A dictionary containing the key in-memory results and saved file paths.
    """

    if config.portfolio_builder not in portfolio_library.PORTFOLIO_BUILDERS:
        raise ValueError(f"Unknown portfolio builder: {config.portfolio_builder}")

    portfolio_builder = portfolio_library.PORTFOLIO_BUILDERS[config.portfolio_builder]
    pf = portfolio_builder(**config.portfolio_kwargs)

    results_dir = _get_results_dir(config, base_results_dir)
    os.makedirs(results_dir, exist_ok=True)

    artifact_paths = _get_expected_artifact_paths(results_dir)

    if not force_rerun and _artifacts_exist(artifact_paths):
        print(f"Skipping {config.name}: existing results found in {results_dir}")
        return {
            "config": config,
            "portfolio": pf,
            "results_dir": results_dir,
            "metadata_path": artifact_paths["metadata_path"],
            "forecast_df": pd.read_csv(artifact_paths["forecast_path"]),
            "realized_df": pd.read_csv(artifact_paths["realized_path"]),
            "backtested_df": pd.read_csv(artifact_paths["backtested_path"]),
            "summary_df": pd.read_csv(artifact_paths["summary_path"]),
            **artifact_paths,
            "skipped": True,
        }

    metadata_path = artifact_paths["metadata_path"]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    forecast_df = forecast.compute_portfolio_var_es_forecast_over_range(
        pf=pf,
        start_date=config.start_date,
        end_date=config.end_date,
        window_size=config.window_size,
        up_to_days=config.up_to_days,
        alpha=config.alpha,
        lambda_val=config.lambda_val,
        num_simulations=config.num_simulations,
    )
    forecast_path = artifact_paths["forecast_path"]
    forecast_df.to_csv(forecast_path, index=False)

    realized_df = backtest.extend_results_with_realized_pnl(
        results_df=forecast_df,
        pf=pf,
    )
    realized_path = artifact_paths["realized_path"]
    realized_df.to_csv(realized_path, index=False)

    backtested_df = backtest.backtest_exceeded_var_es_portfolio(realized_df)
    backtested_path = artifact_paths["backtested_path"]
    backtested_df.to_csv(backtested_path, index=False)

    summary_df = analysis.summarize_backtest_results(
        backtested_df,
        output_path=artifact_paths["summary_path"],
    )

    exceedance_windows_path = analysis.plot_backtest_exceedance_windows(
        backtested_df,
        output_path=artifact_paths["exceedance_windows_path"],
    )
    exceedance_scatter_path = analysis.plot_backtest_exceedance_scatter(
        backtested_df,
        output_path=artifact_paths["exceedance_scatter_path"],
    )
    hit_sequence_path = analysis.plot_backtest_hit_sequence(
        backtested_df,
        output_path=artifact_paths["hit_sequence_path"],
    )
    realized_loss_vs_threshold_path = analysis.plot_backtest_realized_loss_vs_threshold(
        backtested_df,
        output_path=artifact_paths["realized_loss_vs_threshold_path"],
    )

    return {
        "config": config,
        "portfolio": pf,
        "results_dir": results_dir,
        "metadata_path": metadata_path,
        "forecast_df": forecast_df,
        "realized_df": realized_df,
        "backtested_df": backtested_df,
        "summary_df": summary_df,
        "forecast_path": forecast_path,
        "realized_path": realized_path,
        "backtested_path": backtested_path,
        "summary_path": artifact_paths["summary_path"],
        "exceedance_windows_path": exceedance_windows_path,
        "exceedance_scatter_path": exceedance_scatter_path,
        "hit_sequence_path": hit_sequence_path,
        "realized_loss_vs_threshold_path": realized_loss_vs_threshold_path,
        "skipped": False,
    }
