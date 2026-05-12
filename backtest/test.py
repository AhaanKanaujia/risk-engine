import math
import numpy as np
from scipy.stats import chi2, binom, binomtest

def kupiec_proportion_of_failures_test(
    num_observations: int,
    num_exceedances: int,
    alpha: float,
) -> tuple[float, float]:
    """
    Compute the Kupiec Proportion of Failures (POF) test statistic and p-value.

    Args:
        num_observations: The number of backtest observations.
        num_exceedances: The number of observed exceedances.
        alpha: The confidence level used in the VaR/ES forecast.

    Returns:
        A tuple containing the likelihood ratio statistic and the p-value.
    """

    if num_observations <= 0:
        return np.nan, np.nan

    expected_failure_probability = 1 - alpha
    observed_failure_probability = num_exceedances / num_observations

    if expected_failure_probability <= 0 or expected_failure_probability >= 1:
        return np.nan, np.nan

    if observed_failure_probability <= 0:
        observed_failure_probability = 1e-12
    elif observed_failure_probability >= 1:
        observed_failure_probability = 1 - 1e-12

    log_likelihood_null = (
        (num_observations - num_exceedances) * math.log(1 - expected_failure_probability)
        + num_exceedances * math.log(expected_failure_probability)
    )
    log_likelihood_alternative = (
        (num_observations - num_exceedances) * math.log(1 - observed_failure_probability)
        + num_exceedances * math.log(observed_failure_probability)
    )

    lr_stat = -2 * (log_likelihood_null - log_likelihood_alternative)
    p_value = 1 - chi2.cdf(lr_stat, df=1)

    return float(lr_stat), float(p_value)

def exact_binomial_coverage_test(
    num_observations: int,
    num_exceedances: int,
    alpha: float,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute an exact binomial p-value and confidence interval for the exceedance rate.

    Args:
        num_observations: The number of backtest observations.
        num_exceedances: The number of observed exceedances.
        alpha: The confidence level used in the VaR/ES forecast.
        confidence_level: The confidence level for the exact binomial interval.

    Returns:
        A tuple containing:
        - exact binomial p-value for the null exceedance probability of 1 - alpha
        - lower bound of the exact confidence interval for the observed exceedance rate
        - upper bound of the exact confidence interval for the observed exceedance rate
    """

    if num_observations <= 0:
        return np.nan, np.nan, np.nan

    expected_failure_probability = 1 - alpha
    if expected_failure_probability <= 0 or expected_failure_probability >= 1:
        return np.nan, np.nan, np.nan

    test_result = binomtest(
        k=num_exceedances,
        n=num_observations,
        p=expected_failure_probability,
        alternative="two-sided",
    )
    confidence_interval = test_result.proportion_ci(
        confidence_level=confidence_level,
        method="exact",
    )

    return (
        float(test_result.pvalue),
        float(confidence_interval.low),
        float(confidence_interval.high),
    )

def basel_traffic_light_classification(
    num_observations: int,
    num_exceedances: int,
    alpha: float,
) -> tuple[str, int, int]:
    """
    Compute a Basel-style traffic-light classification using binomial exceedance thresholds.

    Args:
        num_observations: The number of backtest observations.
        num_exceedances: The number of observed exceedances.
        alpha: The confidence level used in the VaR/ES forecast.

    Returns:
        A tuple containing:
        - traffic-light classification ('green', 'yellow', or 'red')
        - upper exceedance count threshold for the green region
        - upper exceedance count threshold for the yellow region
    """

    if num_observations <= 0:
        return "n/a", -1, -1

    expected_failure_probability = 1 - alpha
    if expected_failure_probability <= 0 or expected_failure_probability >= 1:
        return "n/a", -1, -1

    green_upper = int(binom.ppf(0.95, num_observations, expected_failure_probability))
    yellow_upper = int(binom.ppf(0.9999, num_observations, expected_failure_probability))

    if num_exceedances <= green_upper:
        classification = "green"
    elif num_exceedances <= yellow_upper:
        classification = "yellow"
    else:
        classification = "red"

    return classification, green_upper, yellow_upper
