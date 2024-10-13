"""Module for exemplar selectors."""

from promptolution.exemplar_selectors.random_search_selector import RandomSearchSelector
from promptolution.exemplar_selectors.random_selector import RandomSelector
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask


def get_exemplar_selector(name: str, task: BaseTask, predictor: BasePredictor):
    """Factory function to get an exemplar selector based on the given name.

    Args:
        name (str): The name of the exemplar selector to instantiate.
        task (BaseTask): The task object to be passed to the selector.
        predictor (BasePredictor): The predictor object to be passed to the selector.

    Returns:
        BaseExemplarSelector: An instance of the requested exemplar selector.

    Raises:
        ValueError: If the requested selector name is not found in the SELECTOR_MAPPING.
    """
    if name == "random":
        return RandomSelector(task, predictor)

    if name == "random_search":
        return RandomSearchSelector(task, predictor)

    raise ValueError(f"Unknown exemplar selector: {name}")
