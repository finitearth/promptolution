"""Mock predictor for testing purposes."""
from typing import List, Optional, Tuple

import numpy as np

from promptolution import BaseLLM, BasePredictor


class MockPredictor(BasePredictor):
    """Mock predictor for testing purposes.

    This class allows precise control over prediction behavior for testing
    without loading actual models or running real inference.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        classes: List[str] = None,
        predetermined_predictions: Optional[dict] = None,
        *args,
        **kwargs
    ):
        """Initialize the MockPredictor.

        Args:
            llm: Language model to use (can be a MockLLM)
            classes: List of possible class labels
            predetermined_predictions: Dictionary mapping inputs to predictions
            *args, **kwargs: Additional arguments to pass to BasePredictor
        """
        super().__init__(llm=llm, *args, **kwargs)
        self.classes = classes or ["neutral", "positive", "negative"]
        self.predetermined_predictions = predetermined_predictions or {}
        self.call_history = []

    def _extract_preds(self, preds: List[str], shape: Tuple[int, int] = None) -> np.ndarray:
        """Extract predictions based on predetermined mapping or default behavior.

        Args:
            preds: Raw text predictions
            shape: Shape for reshaping results (optional)

        Returns:
            np.ndarray: Extracted predictions
        """
        # Record call for test assertions
        self.call_history.append({"preds": preds, "shape": shape})

        results = []
        for pred in preds:
            if pred in self.predetermined_predictions:
                results.append(self.predetermined_predictions[pred])
            else:
                # Default to first class if no match
                results.append(self.classes[0])

        results_array = np.array(results)

        # Reshape if shape is provided
        if shape is not None:
            results_array = results_array.reshape(shape)

        return results_array
