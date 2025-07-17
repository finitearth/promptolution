"""Module for Reward tasks."""


import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Callable, Literal, Optional

from promptolution.tasks.base_task import BaseTask

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.config import ExperimentConfig


class RewardTask(BaseTask):
    """A task that evaluates a predictor using a reward function.

    This task takes a DataFrame, a column name for input data, and a reward function.
    The reward function should take a prediction and its corresponding ground truth (if any), and return a reward.
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
        """Initialize the RewardTask."""
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

    def _calculate_score(self, x: np.ndarray, _: np.ndarray, pred: np.ndarray, **kwargs) -> float:
        """Calculate the score for a single reward prediction using the reward function."""
        return self.reward_function(pred)
