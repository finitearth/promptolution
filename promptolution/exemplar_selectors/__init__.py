"""Module for exemplar selectors."""

from typing import Literal

from promptolution.exemplar_selectors.random_search_selector import RandomSearchSelector
from promptolution.exemplar_selectors.random_selector import RandomSelector
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask

SELECTOR_MAP = {
    "random": RandomSelector,
    "random_search": RandomSearchSelector,
}


def get_exemplar_selector(name: Literal["random", "random_search"], task: BaseTask, predictor: BasePredictor):
    """Factory function to get an exemplar selector based on the given name.

    Args:
        name (str): The name of the exemplar selector to instantiate.
        task (BaseTask): The task object to be passed to the selector.
        predictor (BasePredictor): The predictor object to be passed to the selector.

    Returns:
        BaseExemplarSelector: An instance of the requested exemplar selector.

    Raises:
        ValueError: If the requested selector name is not found.
    """
    if name not in SELECTOR_MAP:
        raise ValueError(f"Exemplar selector '{name}' not found. Available selectors: {list(SELECTOR_MAP.keys())}")

    return SELECTOR_MAP[name](task, predictor)
