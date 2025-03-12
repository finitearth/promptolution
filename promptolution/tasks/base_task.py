"""Base module for tasks."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseTask(ABC):
    """Abstract base class for tasks in the promptolution library.

    This class defines the interface that all concrete task implementations should follow.

    Methods:
        evaluate: An abstract method that should be implemented by subclasses
                  to evaluate prompts using a given predictor.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the BaseTask."""
        pass

    @abstractmethod
    def evaluate(self, prompts: List[str], predictor) -> np.ndarray:
        """Abstract method to evaluate prompts using a given predictor.

        Args:
            prompts (List[str]): List of prompts to evaluate.
            predictor: The predictor to use for evaluation.

        Returns:
            np.ndarray: Array of evaluation scores for each prompt.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError


class DummyTask(BaseTask):
    """A dummy task implementation for testing purposes.

    This task generates random evaluation scores for given prompts.

    Attributes:
        initial_population (List[str]): List of initial prompts.
        description (str): Description of the dummy task.
        xs (np.ndarray): Array of dummy input data.
        ys (np.ndarray): Array of dummy labels.
        classes (List[str]): List of possible class labels.
    """

    def __init__(self):
        """Initialize the DummyTask."""
        self.initial_population = ["Some", "initial", "prompts", "that", "will", "do", "the", "trick"]
        self.description = "This is a dummy task for testing purposes."
        self.xs = np.array(["This is a test", "This is another test", "This is a third test"])
        self.ys = np.array(["positive", "negative", "positive"])
        self.classes = ["negative", "positive"]

    def evaluate(self, prompts: List[str], predictor) -> np.ndarray:
        """Generate random evaluation scores for the given prompts.

        Args:
            prompts (List[str]): List of prompts to evaluate.
            predictor: The predictor to use for evaluation (ignored in this implementation).

        Returns:
            np.ndarray: Array of random evaluation scores, one for each prompt.
        """
        return np.array([np.random.rand()] * len(prompts))
