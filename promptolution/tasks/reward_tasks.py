"""Module for Reward tasks."""


import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Callable, Optional

from promptolution.tasks.base_task import BaseTask

if TYPE_CHECKING:
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.utils.config import ExperimentConfig


class RewardTask(BaseTask):
    """A task that evaluates a predictor using a reward function.

    This task takes a DataFrame, a column name for input data, and a reward function.
    The reward function should take predictions and return rewards.
    """

    def __init__(self, df: pd.DataFrame, x_column: str, reward_function: Callable, config: "ExperimentConfig" = None):
        """Initialize the RewardTask."""
        self.reward_function = reward_function
        self.df = df
        self.x_column = x_column
        super().__init__(config)

    def evaluate(self, predictor: "BasePredictor", **kwargs) -> float:
        """Evaluate the predictor on the dataset using the reward function.

        Args:
            predictor (BasePredictor): The predictor to evaluate.
            **kwargs: Additional arguments for the reward function.

        Returns:
            float: The mean of the rewards.
        """
        predictions = predictor.predict(self.df[self.x_column].tolist(), **kwargs)
        rewards = self.reward_function(predictions, **kwargs)
        return np.mean(rewards)
