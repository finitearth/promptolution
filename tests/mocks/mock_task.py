"""Mock task for testing purposes."""
from typing import List
from unittest.mock import MagicMock

import numpy as np

from promptolution.tasks.base_task import BaseTask


class MockTask(BaseTask):
    """Mock task for testing optimizers.

    This class simulates a classification task without requiring
    actual data or model inference.
    """

    def __init__(self, predetermined_scores=None):
        """Initialize the MockTask with optional predetermined scores.

        Args:
            predetermined_scores: Dictionary mapping prompts to scores,
                or a list of scores to return in sequence, or a function
                that generates scores based on prompts.
        """
        super().__init__()
        self.predetermined_scores = predetermined_scores or {}
        self.call_history = []
        self.score_index = 0

        self.x_column = "x"
        self.y_column = "y"
        # Default attributes similar to ClassificationTask
        self.description = "Mock classification task"
        self.classes = ["positive", "neutral", "negative"]
        self.xs = np.array(["Sample text 1", "Sample text 2", "Sample text 3"])
        self.ys = np.array(["positive", "negative", "neutral"])
        self.initial_prompts = ["Classify:", "Determine:"]
        self.n_blocks = 10

        self.increment_blocks = MagicMock()
        self.reset_blocks = MagicMock()

    def evaluate(
        self,
        prompts: List[str],
        predictor,
        strategy: str = "subsample",
        system_prompts: List[str] = None,
        return_agg_scores: bool = False,
        return_seq: bool = False,
    ) -> np.ndarray:
        """Evaluate prompts with predetermined scores.

        Args:
            prompts: List of prompts to evaluate
            predictor: Predictor (ignored in mock)
            system_prompts: System prompts (ignored in mock)
            subsample: Whether to subsample (ignored in mock)
            n_samples: Number of samples (ignored in mock)
            return_seq: Whether to return sequences

        Returns:
            np.ndarray of scores, and optionally sequences
        """
        # Record the call
        self.call_history.append(
            {
                "prompts": prompts,
                "predictor": predictor,
                "system_prompts": system_prompts,
                "strategy": strategy,
                "return_agg_scores": return_agg_scores,
                "return_seq": return_seq,
            }
        )

        scores = []
        for prompt in prompts:
            # Handle different types of predetermined_scores
            if callable(self.predetermined_scores):
                # If it's a function, call it with the prompt
                score = self.predetermined_scores(prompt)
            elif isinstance(self.predetermined_scores, dict) and prompt in self.predetermined_scores:
                # If it's a dict, look up the prompt
                score = self.predetermined_scores[prompt]
            elif isinstance(self.predetermined_scores, list):
                # If it's a list, return items in sequence (cycling if needed)
                if self.score_index < len(self.predetermined_scores):
                    score = self.predetermined_scores[self.score_index]
                    self.score_index = (self.score_index + 1) % len(self.predetermined_scores)
                else:
                    score = 0.5  # Default score
            else:
                # Generate a somewhat predictable score based on prompt length
                # (longer prompts get slightly higher scores)
                score = 0.5 + 0.01 * (len(prompt) % 10)

            scores.append(score)

        scores_array = np.array(scores)

        if return_seq:
            # Generate dummy sequences
            seqs = [[f"Input: {x}\nOutput: {prompt}" for x in self.xs] for prompt in prompts]
            return scores_array, seqs

        return scores_array
