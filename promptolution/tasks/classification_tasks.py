"""Module for classification tasks."""


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from typing import TYPE_CHECKING, Any, Callable, List, Literal, Tuple, Union

from promptolution.tasks.base_task import BaseTask

if TYPE_CHECKING:
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.utils.config import ExperimentConfig


class ClassificationTask(BaseTask):
    """A class representing a classification task in the promptolution library.

    This class handles the loading and management of classification datasets,
    as well as the evaluation of predictors on these datasets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        task_description: str = None,
        x_column: str = "x",
        y_column: str = "y",
        n_subsamples: int = 30,
        eval_strategy: Literal["full", "subsample", "sequential_block", "random_block"] = "full",
        seed: int = 42,
        metric: Callable = accuracy_score,
        config: "ExperimentConfig" = None,
    ):
        """Initialize the ClassificationTask from a pandas DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data
            task_description (str): Description of the task
            x_column (str, optional): Name of the column containing input texts. Defaults to "x".
            y_column (str, optional): Name of the column containing labels. Defaults to "y".
            n_subsamples (int, optional): Number of subsamples to use. No subsampling if None. Defaults to None.
            eval_strategy (str, optional): Subsampling strategy to use. Options:
                - "full": Uses the entire dataset for evaluation.
                - "evaluated": Uses only previously evaluated datapoints from the cache.
                - "subsample": Randomly selects n_subsamples datapoints without replacement.
                - "sequential_block": Uses a block of block_size consecutive datapoints, advancing through blocks sequentially.
                - "random_block": Randomly selects a block of block_size consecutive datapoints.
                Defaults to "full".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            metric (Callable, optional): Metric to use for evaluation. Defaults to accuracy_score.
            config (ExperimentConfig, optional): Configuration for the task, overriding defaults.
        """
        self.metric = metric
        super().__init__(
            df=df,
            x_column=x_column,
            y_column=y_column,
            task_description=task_description,
            n_subsamples=n_subsamples,
            eval_strategy=eval_strategy,
            seed=seed,
            config=config,
        )
        self.ys = df[self.y_column].str.lower().values  # Ensure y values are lowercase for consistent comparison
        self.classes = np.unique(self.ys)

    def _calculate_score(self, x: np.ndarray, y: np.ndarray, pred: np.ndarray) -> float:
        """Calculate the score for a single prediction."""
        return self.metric([y], [pred])
