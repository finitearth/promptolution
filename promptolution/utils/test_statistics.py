"""
Implementation of statistical significance tests used in the racing algorithm.
Contains paired t-test functionality to compare prompt performance and determine statistical significance between candidates.
"""

import numpy as np
from scipy.stats import ttest_rel

from typing import Any, Callable, List, Literal

TestStatistics = Literal["paired_t_test"]


def get_test_statistic_func(name: TestStatistics) -> Callable[..., bool]:
    """
    Get the test statistic function based on the name provided.

    Args:
        name (str): Name of the test statistic function.

    Returns:
        callable: The corresponding test statistic function.
    """
    if name == "paired_t_test":
        return paired_t_test
    else:
        raise ValueError(f"Unknown test statistic function: {name}. Should be one of {TestStatistics.__args__}.")


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> bool:
    """
    Uses a paired t-test to test if candidate A's accuracy is significantly
    higher than candidate B's accuracy within a confidence interval of 1-\alpha.
    Assumptions:
    - The samples are paired.
    - The differences between the pairs are normally distributed (-> n > 30).

    Parameters:
        scores_a (List[float]): Array of accuracy scores for candidate A.
        scores_b (List[float]): Array of accuracy scores for candidate B.
        alpha (float): Significance level (default 0.05 for 95% confidence).

    Returns:
        bool: True if candidate A is significantly better than candidate B, False otherwise.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    _, p_value = ttest_rel(scores_a, scores_b, alternative="greater")

    result = p_value < alpha

    return bool(result)
