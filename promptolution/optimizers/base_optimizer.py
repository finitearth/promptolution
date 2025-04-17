"""Base module for optimizers in the promptolution library."""

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from promptolution.tasks.base_task import BaseTask


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings.

    This class defines the configuration parameters for optimizers.

    Attributes:
        optimizer_name (str): Name of the optimizer.
        n_steps (int): Number of optimization steps.
        population_size (int): Size of the population to maintain.
        random_seed (int): Random seed for reproducibility.
        log_path (Optional[str]): Path to save optimization logs.
        n_eval_samples (int): Number of samples to use for evaluation.
    """

    optimizer_name: str = ""
    n_steps: int = 10
    population_size: int = 10
    random_seed: int = 42
    log_path: Optional[str] = None
    n_eval_samples: int = 20
    callbacks: List[str] = field(default_factory=list)


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers.

    This class defines the basic structure and interface for prompt optimization algorithms.
    It follows the Hugging Face-style interface pattern while maintaining compatibility
    with the existing API.

    Attributes:
        config (OptimizerConfig): Configuration for the optimizer.
        prompts (List[str]): List of current prompts being optimized.
        task (BaseTask): The task object used for evaluating prompts.
        callbacks (List[Callable]): List of callback functions to be called during optimization.
        predictor: The predictor used for prompt evaluation (if applicable).
    """

    config_class = OptimizerConfig

    def __init__(
        self,
        initial_prompts: List[str] = None,
        task: BaseTask = None,
        callbacks: List[Callable] = None,
        predictor=None,
        config: Optional[Union[Dict[str, Any], OptimizerConfig]] = None,
        **kwargs
    ):
        """Initialize the optimizer with a configuration and/or direct parameters.

        This constructor supports both the new config-based initialization and
        the legacy parameter-based initialization.

        Args:
            initial_prompts: Initial set of prompts to start optimization with.
            task: Task object for prompt evaluation.
            callbacks: List of callback functions.
            predictor: Predictor for prompt evaluation.
            config: Configuration for the optimizer.
            **kwargs: Additional parameters, passed to config if config is provided.
        """
        # Initialize config
        if config is None:
            config = {}

        if isinstance(config, dict):
            # Merge kwargs into config
            for k, v in kwargs.items():
                config[k] = v
            self.config = self.config_class(**config)
        else:
            self.config = config
            # Override config with kwargs
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        # Set up optimizer state
        self.prompts = initial_prompts or []
        self.task = task
        self.callbacks = callbacks or []
        self.predictor = predictor
        self.n_eval_samples = kwargs.get("n_eval_samples", self.config.n_eval_samples)

        # Set random seed
        np.random.seed(self.config.random_seed)

    @abstractmethod
    def optimize(self, n_steps: Optional[int] = None) -> List[str]:
        """Perform the optimization process.

        This method should be implemented by concrete optimizer classes to define
        the specific optimization algorithm.

        Args:
            n_steps: Number of optimization steps to perform. If None, uses the value from config.

        Returns:
            The optimized list of prompts after all steps.
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
