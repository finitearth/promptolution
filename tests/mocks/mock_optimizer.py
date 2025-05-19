"""Mock Optimizer for Testing."""


from unittest.mock import MagicMock

from promptolution.optimizers.base_optimizer import BaseOptimizer


class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing callbacks and other components.

    This class simulates an optimizer without requiring actual optimization processes.
    """

    def __init__(self, prompts=None, scores=None):
        """Initialize the MockOptimizer with optional prompts and scores.

        Args:
            prompts: List of prompts to use (defaults to sample prompts)
            scores: List of scores to use (defaults to sample scores)
        """
        self.prompts = prompts or ["Sample prompt 1", "Sample prompt 2", "Sample prompt 3"]
        self.scores = scores or [0.8, 0.7, 0.6]

        # Create mock LLMs
        self.meta_llm = MagicMock()
        self.meta_llm.input_token_count = 100
        self.meta_llm.output_token_count = 50

        # Create mock predictor
        self.predictor = MagicMock()
        self.predictor.llm = MagicMock()

        # Set up token counting
        self.token_counts = {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}
        self.predictor.llm.get_token_count = MagicMock(return_value=self.token_counts)

    def set_token_counts(self, input_tokens=None, output_tokens=None, total_tokens=None):
        """Set custom token counts for testing.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total number of tokens
        """
        if input_tokens is not None:
            self.token_counts["input_tokens"] = input_tokens
        if output_tokens is not None:
            self.token_counts["output_tokens"] = output_tokens
        if total_tokens is not None:
            self.token_counts["total_tokens"] = total_tokens

        # Update the mock method
        self.predictor.llm.get_token_count = MagicMock(return_value=self.token_counts)
