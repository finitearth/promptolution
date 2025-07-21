"""Base module for tasks."""


from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.utils.config import ExperimentConfig


class BaseTask(ABC):
    """Abstract base class for tasks in the promptolution library."""

    def __init__(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: Optional[str] = None,
        task_description: str = None,
        n_subsamples: int = 30,
        eval_strategy: Literal["full", "subsample", "sequential_block", "random_block", "evaluated"] = "full",
        seed: int = 42,
        config: "ExperimentConfig" = None,
    ):
        """Initialize the BaseTask.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            x_column (str): Name of the column containing input texts.
            y_column (Optional[str]): Name of the column containing labels/ground truth (if applicable).
            task_description (str): Description of the task.
            n_subsamples (int): Number of subsamples to use for evaluation.
            eval_strategy (Literal): Subsampling strategy ("full", "subsample", "sequential_block", "random_block", "evaluated").
            seed (int): Random seed for reproducibility.
            config (ExperimentConfig, optional): Configuration for the task, overriding defaults.
        """
        self.df = df
        self.x_column = x_column
        self.y_column = y_column
        self.task_description = task_description
        self.n_subsamples = n_subsamples
        self.eval_strategy = eval_strategy
        self.seed = seed

        super().__init__()
        if config is not None:
            config.apply_to(self)

        self.xs = df[self.x_column].values
        self.has_y = y_column is not None
        if self.has_y:
            self.ys = df[self.y_column].values
        else:
            # If no y_column is provided, create a dummy y array
            self.ys = np.array([None] * len(self.xs))

        self.block_idx = 0
        self.n_blocks = len(self.xs) // self.n_subsamples if self.n_subsamples > 0 else 1
        self.rng = np.random.default_rng(seed)

        self.eval_cache = {}  # (prompt, x, y): scores per datapoint
        self.seq_cache = {}  # (prompt, x, y): generating sequence per datapoint

    def subsample(
        self, eval_strategy: Literal["full", "subsample", "sequential_block", "random_block"] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample the dataset based on the specified parameters.

        Args:
            eval_strategy (str, optional): Subsampling strategy to use instead of self.eval_strategy. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Subsampled input data and labels.
        """
        if eval_strategy is None:
            eval_strategy = self.eval_strategy

        if eval_strategy in ["full", "evaluated"]:
            return self.xs, self.ys
        elif eval_strategy == "subsample":
            indices = self.rng.choice(len(self.xs), min(self.n_subsamples, len(self.xs)), replace=False)
            return self.xs[indices], self.ys[indices]
        elif eval_strategy == "random_block":
            block_id = self.rng.integers(0, self.n_blocks)
            start_idx = block_id * self.n_subsamples
            end_idx = min((block_id + 1) * self.n_subsamples, len(self.xs))
            indices = np.arange(start_idx, end_idx)
            return self.xs[indices], self.ys[indices]
        elif eval_strategy == "sequential_block":
            start_idx = self.block_idx * self.n_subsamples
            end_idx = min((self.block_idx + 1) * self.n_subsamples, len(self.xs))
            indices = np.arange(start_idx, end_idx)
            return self.xs[indices], self.ys[indices]
        else:
            raise ValueError(f"Unknown subsampling strategy: '{eval_strategy}'")

    def _prepare_batch(
        self, prompts: List[str], xs: np.ndarray, ys: np.ndarray, eval_strategy: str
    ) -> List[Tuple[str, str, Any]]:
        """Generates (prompt, x, y) keys that require prediction.

        Returns keys not found in eval_cache.
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
            datapoint_scores = []
            datapoint_seqs = []
            for x, y in zip(xs, ys):
                cache_key = (prompt, x, y)
                datapoint_scores.append(self.eval_cache.get(cache_key, np.nan))
                datapoint_seqs.append(self.seq_cache.get(cache_key))
            scores.append(datapoint_scores)
            seqs.append(datapoint_seqs)

        if return_agg_scores:
            scores = [np.nanmean(s) for s in scores]

        scores = np.array(scores)
        seqs = np.array(seqs)

        return scores if not return_seq else (scores, seqs)

    @abstractmethod
    def _evaluate(self, xs: np.ndarray, ys: np.ndarray, preds: np.ndarray) -> List[float]:
        """Abstract method to calculate the score for a predictions.

        This method should be implemented by subclasses based on their specific evaluation logic.
        """
        raise NotImplementedError

    def evaluate(
        self,
        prompts: Union[str, List[str]],
        predictor: "BasePredictor",
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
        batches = self._prepare_batch(prompts, xs, ys, eval_strategy=eval_strategy)
        (prompts_to_evaluate, xs_to_evaluate, ys_to_evaluate) = zip(*batches) if batches else ([], [], [])

        preds = predictor.predict(
            prompts=prompts_to_evaluate,
            xs=xs_to_evaluate,
            system_prompts=system_prompts,
            return_seq=return_seq,
        )

        if return_seq:
            preds, seqs = preds
        scores = self._evaluate(xs_to_evaluate, ys_to_evaluate, preds)
        for i, cache_key in enumerate(batches):
            self.eval_cache[cache_key] = scores[i]

            if return_seq:
                self.seq_cache[cache_key] = seqs[i]

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

        popped_xs = self.xs[indices]
        popped_ys = self.ys[indices]
        df_popped = pd.DataFrame({self.x_column: popped_xs, self.y_column: popped_ys})

        self.xs = np.delete(self.xs, indices)
        self.ys = np.delete(self.ys, indices)

        # Update n_blocks and block_idx based on the new dataset size
        self.n_blocks = len(self.xs) // self.n_subsamples if self.n_subsamples > 0 else 1
        self.block_idx = min(self.block_idx, self.n_blocks - 1) if self.n_blocks > 0 else 0

        # Clear cache for popped items (optional, but good practice if memory is a concern)
        keys_to_remove = []
        for key in self.eval_cache:
            if key[1] in popped_xs and key[2] in popped_ys:  # Check if the x and y correspond to popped data
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.eval_cache.pop(key, None)
            self.seq_cache.pop(key, None)

        return df_popped

    def increment_block_idx(self) -> None:
        """Increment the block index for subsampling.

        Raises:
            ValueError: If the eval_strategy does not contain "block".
        """
        if "block" not in self.eval_strategy:
            raise ValueError("Block increment is only valid for block subsampling.")
        self.block_idx += 1
        if self.n_blocks > 0:  # Ensure n_blocks is not zero to avoid division by zero
            self.block_idx %= self.n_blocks
        else:
            self.block_idx = 0  # If no blocks, reset to 0

    def reset_block_idx(self) -> None:
        """Reset the block index for subsampling.

        Raises:
            ValueError: If the eval_strategy does not contain "block".
        """
        if "block" not in self.eval_strategy:
            raise ValueError("Block reset is only valid for block subsampling.")
        self.block_idx = 0
