import os
import shutil

from reporting import experiment_configs, run_experiment

def run_all_experiments(
    configs: list[experiment_configs.ExperimentConfig] | None = None,
    base_results_dir: str = "./results",
    clear_previous_results: bool = False,
    force_rerun: bool = False,
) -> list[dict[str, object]]:
    """
    Run a list of configured reporting experiments.

    Args:
        configs: Optional list of experiment configs. Defaults to the built-in experiment set.
        base_results_dir: The root results directory.
        clear_previous_results: Whether to remove the existing results directory before running.
        force_rerun: Whether to recompute scenarios even when saved results already exist.

    Returns:
        A list of experiment result dictionaries.
    """

    if configs is None:
        configs = experiment_configs.DEFAULT_EXPERIMENTS

    if clear_previous_results and os.path.isdir(base_results_dir):
        shutil.rmtree(base_results_dir)

    os.makedirs(base_results_dir, exist_ok=True)

    experiment_outputs = []
    for i, config in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] Running reporting experiment: {config.name}")
        experiment_outputs.append(
            run_experiment.run_experiment(
                config=config,
                base_results_dir=base_results_dir,
                force_rerun=force_rerun,
            )
        )

    return experiment_outputs
