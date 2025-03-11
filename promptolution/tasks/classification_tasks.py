"""Module for classification tasks."""

from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask


class ClassificationTask(BaseTask):
    """A class representing a classification task in the promptolution library.

    This class handles the loading and management of classification datasets,
    as well as the evaluation of predictors on these datasets.

    Attributes:
        description (str): Description of the task.
        classes (List[str]): List of possible class labels.
        xs (np.ndarray): Array of input data.
        ys (np.ndarray): Array of labels.
        metric (Callable): Metric to use for evaluation.

    Inherits from:
        BaseTask: The base class for tasks in the promptolution library.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        description: str,
        x_column: str = "x",
        y_column: str = "y",
        metric: Callable = accuracy_score,
    ):
        """Initialize the ClassificationTask from a pandas DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data
            description (str): Description of the task
            x_column (str, optional): Name of the column containing input texts. Defaults to "x".
            y_column (str, optional): Name of the column containing labels. Defaults to "y".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            metric (Callable, optional): Metric to use for evaluation. Defaults to accuracy_score.
        """
        super().__init__()
        self.description = description
        self.metric = metric

        # Sort classes by frequency
        self.classes = df[y_column].unique()

        # Set data attributes
        self.xs = df[x_column].values
        self.ys = df[y_column].values

    def evaluate(
        self,
        prompts: List[str],
        predictor: BasePredictor,
        n_samples: int = 20,
        subsample: bool = False,
        return_seq: bool = False,
    ) -> np.ndarray:
        """Evaluate a set of prompts using a given predictor.

        Args:
            prompts (List[str]): List of prompts to evaluate.
            predictor (BasePredictor): Predictor to use for evaluation.
            n_samples (int, optional): Number of samples to use if subsampling. Defaults to 20.
            subsample (bool, optional): Whether to use subsampling.
            If set to true, samples a different subset per call. Defaults to False.
            return_seq (bool, optional): whether to return the generating sequence
            subsample (bool, optional): Whether to use subsampling.
            If set to true, samples a different subset per call. Defaults to False.
            return_seq (bool, optional): whether to return the generating sequence

        Returns:
            np.ndarray: Array of accuracy scores for each prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        # Randomly select a subsample of n_samples
        if subsample:
            indices = np.random.choice(len(self.xs), n_samples, replace=False)
        else:
            indices = np.arange(len(self.xs))

        xs_subsample = self.xs[indices]
        ys_subsample = self.ys[indices]

        # Make predictions on the subsample
        preds = predictor.predict(prompts, xs_subsample, return_seq=return_seq)

        if return_seq:
            preds, seqs = preds

        scores = np.array([self.metric(ys_subsample, pred) for pred in preds])

        if return_seq:
            return scores, seqs

        return scores
