"""Module for classification tasks."""

from typing import Any, Callable, List, Literal, Tuple, Union

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
    """

    def __init__(
        self,
        df: pd.DataFrame,
        description: str = None,
        x_column: str = "x",
        y_column: str = "y",
        n_subsamples: int = 30,
        block_size: int = 30,
        eval_strategy: Literal["full", "subsample", "sequential_block", "random_block"] = "full",
        seed: int = 42,
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
        self.description = description
        self.metric = metric

        self.x_column = x_column
        self.y_column = y_column

        self.xs = df[x_column].values
        self.ys = df[y_column].str.lower().values
        self.classes = df[y_column].unique()

        self.eval_strategy = eval_strategy
        self.n_subsamples = n_subsamples
        self.block_idx = 0
        self.n_blocks = len(self.xs) // self.n_subsamples
        self.rng = np.random.default_rng(seed)
        super().__init__(config)

        self.eval_cache = {}  # (prompt, x, y): scores per datapoint
        self.seq_cache = {}  # (prompt, x, y): generating sequence per datapoint

    def subsample(
        self, eval_strategy: Literal["full", "subsample", "sequential_block", "random_block"] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample the dataset based on the specified parameters.

        Args:
            strategy (str, optional): Subsampling strategy to use instead of self.subsample_strategy. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Subsampled input data and labels.
        """
        if eval_strategy is None:
            eval_strategy = self.eval_strategy

        if eval_strategy in ["full", "evaluated"]:
            return self.xs, self.ys

        elif eval_strategy == "subsample":
            indices = self.rng.choice(len(self.xs), self.n_subsamples, replace=False)
            return self.xs[indices], self.ys[indices]

        elif eval_strategy == "random_block":
            block_id = self.rng.integers(0, len(self.xs) // self.n_subsamples)
            indices = np.arange(block_id * self.n_subsamples, (block_id + 1) * self.n_subsamples)
            return self.xs[indices], self.ys[indices]

        elif eval_strategy == "sequential_block":
            indices = np.arange(self.block_idx * self.n_subsamples, (self.block_idx + 1) * self.n_subsamples)
            return self.xs[indices], self.ys[indices]

        else:
            raise ValueError(f"Unknown subsampling strategy: '{eval_strategy}")

    def _prepare_batch(
        self, prompts: List[str], xs: np.ndarray, ys: np.ndarray, eval_strategy: str
    ) -> List[Tuple[str, str, str]]:
        """Generates (prompt, x, y) keys that require prediction.

        If strategy is "evaluated", returns an empty list.
        Otherwise, returns keys not found in eval_cache.
        """
        if eval_strategy == "evaluated":
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
        eval_strategy: str = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[List[Any], np.ndarray]]]:
        """Evaluate a set of prompts using a given predictor.

        This method orchestrates subsampling, prediction, caching, and result collection.
        """
        prompts = [prompts] if isinstance(prompts, str) else prompts
        eval_strategy = eval_strategy or self.eval_strategy

        xs, ys = self.subsample(eval_strategy=eval_strategy)
        batches = self._prepare_batch(prompts, xs, ys, eval_strategy)
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
        assert n is None or frac is None, "Only one of n or frac can be specified."
        if n is not None:
            indices = self.rng.choice(len(self.xs), n, replace=False)
        elif frac is not None:
            indices = self.rng.choice(len(self.xs), int(len(self.xs) * frac), replace=False)
        else:
            raise ValueError("Either n or frac must be specified.")

        xs = self.xs[indices]
        ys = self.ys[indices]
        df = pd.DataFrame({self.x_column: xs, self.y_column: ys})

        self.xs = np.delete(self.xs, indices)
        self.ys = np.delete(self.ys, indices)

        self.n_blocks = len(self.xs) // self.n_subsamples
        self.block_idx = min(self.block_idx, self.n_blocks - 1)

        return df

    def increment_block_idx(self) -> None:
        """Increment the block index for subsampling.

        Raises:
            ValueError: If the eval_strategy does not contain "block".
        """
        if "block" not in self.eval_strategy:
            raise ValueError("Block increment is only valid for block subsampling.")
        self.block_idx += 1
        if self.block_idx >= self.n_blocks:
            self.block_idx = 0

    def reset_block_idx(self) -> None:
        """Reset the block index for subsampling.

        Raises:
            ValueError: If the eval_strategy does not contain "block".
        """
        if "block" not in self.eval_strategy:
            raise ValueError("Block reset is only valid for block subsampling.")
        self.block_idx = 0
