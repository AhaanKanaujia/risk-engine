import numpy as np
from scipy.stats import norm

def test_correlation_matrix(simulated_paths, true_corr, alpha=0.05):
    """
    Perform a statistical test to check if the correlation matrix of the simulated log returns matches the true correlation matrix used for calibration.
    
    Args:
        simulated_paths: A numpy array of shape (num_simulations, num_steps, num_stocks) containing the simulated stock price paths.
        true_corr: The true correlation matrix used for calibration.
        alpha: The significance level for the test (default is 0.05).
    
    Returns:
        Dictionary containing results of the test
    """

    n_sim, n_steps_plus1, n = simulated_paths.shape
    N = n_sim * (n_steps_plus1 - 1)

    log_returns = np.diff(np.log(simulated_paths), axis=1)
    data = log_returns.reshape(-1, n)

    sample_corr = np.corrcoef(data.T)

    # create mask for off-diagonal elements
    mask = ~np.eye(n, dtype=bool)

    sample_off = sample_corr[mask]
    true_off = true_corr[mask]

    # Fisher transform only off-diagonal elements
    z_sample = np.arctanh(sample_off)
    z_true = np.arctanh(true_off)

    se = 1 / np.sqrt(N - 3)

    T = (z_sample - z_true) / se

    p_values = 2 * (1 - norm.cdf(np.abs(T)))

    max_abs_error = np.max(np.abs(sample_off - true_off))
    max_test_stat = np.max(np.abs(T))
    min_p_value = np.min(p_values)

    reject = min_p_value < alpha

    return {
        "max_abs_error": max_abs_error,
        "max_test_stat": max_test_stat,
        "min_p_value": min_p_value,
        "reject_null": reject,
        "N_samples": N,
        "expected_error_scale": 1/np.sqrt(N)
    }