"""Base class for prompt optimizers."""

import time
from abc import ABC, abstractmethod
from typing import Callable, List

from promptolution.tasks.base_task import BaseTask


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers.

    This class defines the basic structure and interface for prompt optimization algorithms.
    Concrete optimizer implementations should inherit from this class and implement
    the `optimize` method.

    Attributes:
        prompts (List[str]): List of current prompts being optimized.
        task (BaseTask): The task object used for evaluating prompts.
        callbacks (List[Callable]): List of callback functions to be called during optimization.
        predictor: The predictor used for prompt evaluation (if applicable).

    Args:
        initial_prompts (List[str]): Initial set of prompts to start optimization with.
        task (BaseTask): Task object for prompt evaluation.
        callbacks (List[Callable], optional): List of callback functions. Defaults to an empty list.
        predictor (optional): Predictor for prompt evaluation. Defaults to None.
    """

    def __init__(
        self,
        initial_prompts: list[str],
        task: BaseTask,
        callbacks: list[Callable] = [],
        predictor=None,
        n_eval_samples=20,
    ):
        """Initialize the BaseOptimizer."""
        self.prompts = initial_prompts
        self.task = task
        self.callbacks = callbacks
        self.predictor = predictor
        self.n_eval_samples = n_eval_samples

    @abstractmethod
    def optimize(self, n_steps: int) -> List[str]:
        """Abstract method to perform the optimization process.

        This method should be implemented by concrete optimizer classes to define
        the specific optimization algorithm.

        Args:
            n_steps (int): Number of optimization steps to perform.

        Returns:
            List[str]: The optimized list of prompts after all steps.

        Raises:
            NotImplementedError: If not implemented by a concrete class.
        """
        raise NotImplementedError

    def _on_step_end(self):
        """Call all registered callbacks at the end of each optimization step."""
        continue_optimization = True
        for callback in self.callbacks:
            continue_optimization &= callback.on_step_end(self)  # if any callback returns False, end the optimization

        return continue_optimization

    def _on_epoch_end(self):
        """Call all registered callbacks at the end of each optimization epoch."""
        continue_optimization = True
        for callback in self.callbacks:
            continue_optimization &= callback.on_epoch_end(self)  # if any callback returns False, end the optimization

        return continue_optimization

    def _on_train_end(self):
        """Call all registered callbacks at the end of the entire optimization process."""
        for callback in self.callbacks:
            callback.on_train_end(self)


class DummyOptimizer(BaseOptimizer):
    """A dummy optimizer that doesn't perform any actual optimization.

    This optimizer simply returns the initial prompts without modification.
    It's useful for testing or as a baseline comparison.

    Attributes:
        prompts (List[str]): List of prompts (unchanged from initialization).
        callbacks (List[Callable]): Empty list of callbacks.

    Args:
        initial_prompts (List[str]): Initial set of prompts.
        *args: Variable length argument list (unused).
        **kwargs: Arbitrary keyword arguments (unused).
    """

    def __init__(self, initial_prompts, *args, **kwargs):
        """Initialize the DummyOptimizer."""
        self.callbacks = []
        self.prompts = initial_prompts

    def optimize(self, n_steps) -> list[str]:
        """Simulate an optimization process without actually modifying the prompts.

        This method calls the callback methods to simulate a complete optimization
        cycle, but returns the initial prompts unchanged.

        Args:
            n_steps (int): Number of optimization steps (unused in this implementation).

        Returns:
            List[str]: The original list of prompts, unchanged.
        """
        self._on_step_end()
        self._on_epoch_end()
        self._on_train_end()

        return self.prompts
