"""Module for classification tasks."""

from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from promptolution.config import ExperimentConfig
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
        initial_prompts (List[str]): Initial set of prompts to start optimization with.
        metric (Callable): Metric to use for evaluation.
        config (ExperimentConfig): Configuration for the experiment.

    Inherits from:
        BaseTask: The base class for tasks in the promptolution library.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        description: str = None,
        x_column: str = "x",
        y_column: str = "y",
        n_subsamples: int = 30,
        block_size: int = 30,
        subsample_strategy: str = "full",
        metric: Callable = accuracy_score,
        config: ExperimentConfig = None,
    ):
        """Initialize the ClassificationTask from a pandas DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data
            description (str): Description of the task
            x_column (str, optional): Name of the column containing input texts. Defaults to "x".
            y_column (str, optional): Name of the column containing labels. Defaults to "y".
            n_subsamples (int, optional): Number of subsamples to use. No subsampling if None. Defaults to None.
            block_size (int, optional): Block size for subsampling. Defaults to None.
            subsample_strategy (str, optional): Subsampling strategy to use. Can be "full", "subsample", "sequential_block" or "random_block". Defaults to None.
            metric (Callable, optional): Metric to use for evaluation. Defaults to accuracy_score.
            config (ExperimentConfig, optional): ExperimentConfig overwriting the defaults.
        """
        self.description = description
        self.metric = metric

        self.x_column = x_column
        self.y_column = y_column

        self.xs = df[x_column].values
        self.ys = df[y_column].str.lower().values
        self.classes = df[y_column].unique()

        self.subsample_strategy = subsample_strategy
        self.n_subsamples = n_subsamples
        self.block_size = block_size
        self.block_idx = 0
        self.n_blocks = len(self.xs) // self.block_size
        super().__init__(config)

        self.eval_cache = {}  # (prompt, x, y): scores per datapoint
        self.seq_cache = {}  # (prompt, x, y): generating sequence per datapoint

    def subsample(self, strategy: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample the dataset based on the specified parameters.

        Args:
            strategy (str, optional): Subsampling strategy to use instead of self.subsample_strategy. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Subsampled input data and labels.
        """
        if strategy is None:
            strategy = self.subsample_strategy

        if strategy in ["full", "evaluated"]:
            return self.xs, self.ys

        elif strategy == "subsample":
            indices = np.random.choice(len(self.xs), self.n_subsamples, replace=False)
            return self.xs[indices], self.ys[indices]

        elif strategy == "random_block":
            block_id = np.random.randint(0, len(self.xs) // self.block_size)
            indices = np.arange(block_id * self.block_size, (block_id + 1) * self.block_size)
            return self.xs[indices], self.ys[indices]

        elif strategy == "sequential_block":
            indices = np.arange(self.block_idx * self.block_size, (self.block_idx + 1) * self.block_size)
            return self.xs[indices], self.ys[indices]

        else:
            raise ValueError(f"Unknown subsampling strategy: '{strategy}")

    def _prepare_batch(
        self, prompts: List[str], xs: np.ndarray, ys: np.ndarray, strategy: str
    ) -> List[Tuple[str, str, str]]:
        """Generates (prompt, x, y) keys that require prediction.

        If strategy is "evaluated", returns an empty list.
        Otherwise, returns keys not found in eval_cache.
        """
        if strategy == "evaluated":
            return []

        keys_to_predict = []
        for prompt in prompts:
            for x, y in zip(xs, ys):
                cache_key = (prompt, x, y)
                if cache_key not in self.eval_cache:
                    keys_to_predict.append(cache_key)
        return keys_to_predict

    def _collect_results_from_cache(
        self,
        prompts: List[str],
        xs: np.ndarray,
        ys: np.ndarray,
        return_agg_scores: bool,
        return_seq: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[List[Any], np.ndarray]]]:
        """Collects all results for the current batch from the cache and formats them."""
        scores = []
        seqs = []

        for prompt in prompts:
            cache_keys = [(prompt, x, y) for x, y in zip(xs, ys)]
            scores += [[self.eval_cache.get(key, np.nan) for key in cache_keys]]
            seqs += [[self.seq_cache.get(key) for key in cache_keys]]
        if return_agg_scores:
            scores = [np.nanmean(s) for s in scores]
        scores = np.array(scores)
        seqs = np.array(seqs)

        return scores if not return_seq else (scores, seqs)

    def evaluate(
        self,
        prompts: Union[str, List[str]],
        predictor: BasePredictor,
        system_prompts: List[str] = None,
        return_agg_scores: bool = True,
        return_seq: bool = False,
        strategy: str = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[List[Any], np.ndarray]]]:
        """Evaluate a set of prompts using a given predictor.

        This method orchestrates subsampling, prediction, caching, and result collection.
        """
        prompts = [prompts] if isinstance(prompts, str) else prompts
        strategy = strategy or self.subsample_strategy

        xs, ys = self.subsample(strategy=strategy)
        batches = self._prepare_batch(prompts, xs, ys, strategy)
        prompts_to_evaluate, xs_to_evaluate, ys_to_evaluate = zip(*batches) if batches else ([], [], [])

        preds = predictor.predict(
            prompts=prompts_to_evaluate,
            xs=xs_to_evaluate,
            system_prompts=system_prompts,
            return_seq=return_seq,
        )

        if return_seq:
            preds, seqs = preds

        for i, cache_key in enumerate(batches):
            y_pred, y_true = preds[i], ys_to_evaluate[i]
            if return_seq:
                self.seq_cache[cache_key] = seqs[i]
            self.eval_cache[cache_key] = self.metric([y_pred], [y_true])

        return self._collect_results_from_cache(
            prompts,
            xs,
            ys,
            return_agg_scores,
            return_seq,
        )

    def pop_datapoints(self, n: int = None, frac: float = None) -> pd.DataFrame:
        """Pop a number of datapoints from the dataset.

        Args:
            n (int, optional): Number of datapoints to pop. Defaults to None.
            frac (float, optional): Fraction of datapoints to pop. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the popped datapoints.
        """
        if n is not None:
            indices = np.random.choice(len(self.xs), n, replace=False)
        elif frac is not None:
            indices = np.random.choice(len(self.xs), int(len(self.xs) * frac), replace=False)
        else:
            raise ValueError("Either n or frac must be specified.")

        xs = self.xs[indices]
        ys = self.ys[indices]
        df = pd.DataFrame({self.x_column: xs, self.y_column: ys})

        self.xs = np.delete(self.xs, indices)
        self.ys = np.delete(self.ys, indices)

        self.n_blocks = len(self.xs) // self.block_size
        self.block_idx = min(self.block_idx, self.n_blocks - 1)

        return df

    def increment_blocks(self) -> None:
        """Increment the block index for subsampling."""
        if "block" not in self.subsample_strategy:
            raise ValueError("Block increment is only valid for block subsampling.")
        self.block_idx += 1
        if self.block_idx >= self.n_blocks:
            self.block_idx = 0

    def reset_blocks(self) -> None:
        """Reset the block index for subsampling."""
        if "block" not in self.subsample_strategy:
            raise ValueError("Block reset is only valid for block subsampling.")
        self.block_idx = 0
