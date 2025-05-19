"""Base module for tasks."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from promptolution.utils.config import ExperimentConfig


class BaseTask(ABC):
    """Abstract base class for tasks in the promptolution library.

    This class defines the interface that all concrete task implementations should follow.

    Methods:
        evaluate: An abstract method that should be implemented by subclasses
                  to evaluate prompts using a given predictor.
    """

    def __init__(self, config: "ExperimentConfig" = None):
        """Initialize the BaseTask."""
        if config is not None:
            config.apply_to(self)

    @abstractmethod
    def evaluate(self, prompts: List[str], predictor, system_prompts: List[str] = None) -> np.ndarray:
        """Abstract method to evaluate prompts using a given predictor.

        Args:
            prompts (List[str]): List of prompts to evaluate.
            predictor: The predictor to use for evaluation.
            system_prompts (List[str]): List of system prompts to evaluate.

        Returns:
            np.ndarray: Array of evaluation scores for each prompt.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError
