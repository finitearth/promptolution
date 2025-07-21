"""Module for Reward tasks."""


import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Callable, List, Literal, Optional

from promptolution.tasks.base_task import BaseTask

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.config import ExperimentConfig


class RewardTask(BaseTask):
    """A task that evaluates a predictor using a reward function.

    This task takes a DataFrame, a column name for input data, and a reward function.
    The reward function should take a prediction and return a reward.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        reward_function: Callable,
        x_column: str = "x",
        task_description: Optional[str] = None,
        n_subsamples: int = 30,
        eval_strategy: Literal["full", "subsample", "sequential_block", "random_block"] = "full",
        seed: int = 42,
        config: "ExperimentConfig" = None,
    ):
        """Initialize the RewardTask.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data.
            reward_function (Callable): Function that takes a prediction and returns a reward score.
            x_column (str, optional): Name of the column containing input texts. Defaults to "x".
            task_description (str, optional): Description of the task.
            n_subsamples (int, optional): Number of subsamples to use. Defaults to 30.
            eval_strategy (str, optional): Subsampling strategy to use. Defaults to "full".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            config (ExperimentConfig, optional): Configuration for the task, overriding defaults.
        """
        self.reward_function = reward_function
        super().__init__(
            df=df,
            x_column=x_column,
            task_description=task_description,
            n_subsamples=n_subsamples,
            eval_strategy=eval_strategy,
            seed=seed,
            config=config,
        )

    def _evaluate(self, xs: np.ndarray, ys: np.ndarray, preds: np.ndarray) -> List[float]:
        """Calculate the score for a single reward prediction using the reward function."""
        rewards = [self.reward_function(pred) for pred in preds]
        return rewards
