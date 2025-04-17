"""Base module for tasks in the promptolution library."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class TaskConfig:
    """Configuration for task settings.

    This class defines the configuration parameters for tasks.

    Attributes:
        task_name (str): Name of the task.
        dataset_path (Optional[str]): Path to the dataset.
        dataset_description (Optional[str]): Description of the dataset.
        classes (Optional[List[str]]): List of class labels for classification tasks.
        initial_prompts (Optional[List[str]]): List of initial prompts for the task.
        evaluation_metric (str): Metric used for evaluating prompt performance.
        num_eval_samples (int): Number of samples to use for evaluation.
    """

    task_name: str = ""
    dataset_path: Optional[str] = None
    dataset_description: Optional[str] = None
    classes: List[str] = field(default_factory=list)
    initial_prompts: List[str] = field(default_factory=list)
    evaluation_metric: str = "accuracy"
    num_eval_samples: int = 20


class BaseTask(ABC):
    """Abstract base class for tasks in the promptolution library.

    This class defines the interface that all concrete task implementations should follow.

    Attributes:
        config (TaskConfig): Configuration for the task.
        xs (np.ndarray): Input examples for the task.
        ys (np.ndarray): Labels or target outputs for the task.
        initial_population (List[str]): Initial prompt population for optimization.
    """

    config_class = TaskConfig

    def __init__(self, *args, **kwargs):
        """Initialize the task with a configuration or direct parameters.

        This constructor supports both the new config-based initialization and
        the legacy parameter-based initialization.

        Args:
            *args: Positional arguments (for backward compatibility).
            **kwargs: Keyword arguments either for config fields or direct parameters.
        """
        # Get configuration if provided
        config = kwargs.pop("config", None)

        # Initialize config
        if config is None:
            # Check if first arg is a config
            if args and isinstance(args[0], self.config_class):
                self.config = args[0]
            else:
                # Create config from kwargs
                self.config = self.config_class(**kwargs)
        elif isinstance(config, dict):
            # Create config from dict
            self.config = self.config_class(**config)
        else:
            # Use provided config object
            self.config = config

        # Initialize task properties
        self.xs = np.array([])
        self.ys = np.array([])
        self.initial_population = self.config.initial_prompts or []

    def _load_data(self):
        """Load task data from the dataset path.

        This method should be implemented by subclasses to load task-specific data.
        """
        pass

    @abstractmethod
    def evaluate(self, prompts: List[str], predictor, system_promtps: List[str] = None) -> np.ndarray:
        """Abstract method to evaluate prompts using a given predictor.

        Args:
            prompts: List of prompts to evaluate.
            predictor: The predictor to use for evaluation.
            system_promtps (List[str]): List of system prompts to evaluate.

        Returns:
            np.ndarray: Array of evaluation scores for each prompt.
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

    def evaluate(self, prompts: List[str], predictor, system_prompts=None) -> np.ndarray:
        """Generate random evaluation scores for the given prompts.

        Args:
            prompts (List[str]): List of prompts to evaluate.
            predictor: The predictor to use for evaluation (ignored in this implementation).

        Returns:
            np.ndarray: Array of random evaluation scores, one for each prompt.
        """
        return np.array([np.random.rand()] * len(prompts))
