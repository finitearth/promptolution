"""Base module for optimizers in the promptolution library."""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Callable, List

from promptolution.config import ExperimentConfig
from promptolution.tasks.base_task import BaseTask

logger = getLogger(__name__)


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers.

    This class defines the basic structure and interface for prompt optimization algorithms.

    Attributes:
        config (OptimizerConfig): Configuration for the optimizer.
        prompts (List[str]): List of current prompts being optimized.
        task (BaseTask): The task object used for evaluating prompts.
        callbacks (List[Callable]): List of callback functions to be called during optimization.
        predictor: The predictor used for prompt evaluation (if applicable).
    """

    def __init__(
        self,
        predictor,
        task: BaseTask,
        initial_prompts: List[str],
        callbacks: List[Callable] = None,
        config: ExperimentConfig = None,
    ):
        """Initialize the optimizer with a configuration and/or direct parameters.

        Args:
            initial_prompts: Initial set of prompts to start optimization with.
            task: Task object for prompt evaluation.
            callbacks: List of callback functions.
            predictor: Predictor for prompt evaluation.
            config: Configuration for the optimizer.
        """
        # Set up optimizer state
        self.prompts = initial_prompts
        self.task = task
        self.callbacks = callbacks or []
        self.predictor = predictor
        self.verbosity = 0

        if config is not None:
            config.apply_to(self)

        self.config = config

    def optimize(self, n_steps: int) -> List[str]:
        """Perform the optimization process.

        This method should be implemented by concrete optimizer classes to define
        the specific optimization algorithm.

        Args:
            n_steps: Number of optimization steps to perform. If None, uses the value from config.

        Returns:
            The optimized list of prompts after all steps.
        """
        # validate config
        if self.config is not None:
            self.config.validate()
        self._pre_optimization_loop()

        for _ in range(n_steps):
            try:
                self.prompts = self._step()
            except Exception as e:
                # exit training loop and gracefully fail
                logger.error(f"Error during optimization step: {e}")
                logger.error("Exiting optimization loop.")
                break

            # Callbacks at the end of each step
            continue_optimization = self._on_step_end()
            if not continue_optimization:
                break

        self._on_train_end()

        return self.prompts

    @abstractmethod
    def _pre_optimization_loop(self):
        """Prepare for the optimization loop.

        This method should be implemented by concrete optimizer classes to define
        any setup required before the optimization loop starts.
        """
        pass

    @abstractmethod
    def _step(self):
        """Perform a single optimization step.

        This method should be implemented by concrete optimizer classes to define
        the specific optimization step.

        Returns:
            The optimized list of prompts after the step.
        """
        pass

    def _on_step_end(self):
        """Call all registered callbacks at the end of each optimization step."""
        continue_optimization = True
        for callback in self.callbacks:
            continue_optimization &= callback.on_step_end(self)  # if any callback returns False, end the optimization

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

    def __init__(self, initial_prompts):
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
        self._on_train_end()

        return self.prompts
