"""Wilcoxon signed-rank, Friedman, and Holm post-hoc tests."""

import numpy as np
from scipy import stats


def wilcoxon_test(results_a, results_b, alternative="less"):
    """Paired Wilcoxon signed-rank test; alternative="less" tests A < B.
    Returns (stat, p). A ValueError (all differences zero) counts as p=1."""
    try:
        stat, p = stats.wilcoxon(results_a, results_b, alternative=alternative)
    except ValueError:
        stat, p = 0.0, 1.0
    return stat, p


def friedman_test(results_matrix):
    """Friedman test on an (n_datasets, n_algorithms) matrix of means.
    Returns (chi2 stat, p, average ranks); rank 1 = best."""
    n_datasets, n_algorithms = results_matrix.shape

    rank_matrix = np.zeros_like(results_matrix)
    for i in range(n_datasets):
        rank_matrix[i] = stats.rankdata(results_matrix[i])

    avg_ranks = np.mean(rank_matrix, axis=0)

    stat, p = stats.friedmanchisquare(
        *[results_matrix[:, j] for j in range(n_algorithms)]
    )

    return stat, p, avg_ranks


def holm_posthoc(avg_ranks, n_datasets, control_idx=0):
    """Holm step-down procedure against a control algorithm. Returns a list
    of dicts with algorithm_idx, z_value, p_value, adjusted_alpha,
    significant."""
    k = len(avg_ranks)
    SE = np.sqrt(k * (k + 1) / (6 * n_datasets))

    comparisons = []
    for j in range(k):
        if j == control_idx:
            continue
        z = (avg_ranks[j] - avg_ranks[control_idx]) / SE
        p = 2 * stats.norm.sf(abs(z))  # two-sided
        comparisons.append({"algorithm_idx": j, "z_value": z, "p_value": p})

    comparisons.sort(key=lambda x: x["p_value"])

    m = len(comparisons)
    for rank, comp in enumerate(comparisons):
        adjusted_alpha = 0.05 / (m - rank)
        comp["adjusted_alpha"] = adjusted_alpha
        comp["significant"] = comp["p_value"] < adjusted_alpha

    return comparisons
