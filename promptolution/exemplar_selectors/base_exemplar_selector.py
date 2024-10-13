"""Base class for exemplar selectors."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask


class BaseExemplarSelector(ABC):
    """An abstract base class for exemplar selectors.

    This class defines the basic interface and common functionality
    that all exemplar selectors should implement.
    """

    def __init__(self, task: BaseTask, predictor: BasePredictor):
        """Initialize the BaseExemplarSelector.

        Args:
            task (BaseTask): An object representing the task to be performed.
            predictor (BasePredictor): An object capable of making predictions based on prompts.
        """
        self.task = task
        self.predictor = predictor

    @abstractmethod
    def select_exemplars(self, prompt: str, n_examples: int = 5) -> str:
        """Select exemplars based on the given prompt.

        Args:
            prompt (str): The input prompt to base the exemplar selection on.
            n_examples (int, optional): The number of exemplars to select. Defaults to 5.

        Returns:
            str: A new prompt that includes the original prompt and the selected exemplars.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
