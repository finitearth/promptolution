from abc import abstractmethod
from typing import List

import numpy as np


class BasePredictor:
    """
    Abstract base class for predictors in the promptolution library.

    This class defines the interface that all concrete predictor implementations should follow.

    Attributes:
        model_id (str): Identifier for the model used by the predictor.
        classes (List[str]): List of possible class labels for classification tasks.

    Methods:
        predict: An abstract method that should be implemented by subclasses
                 to make predictions based on prompts and input data.
    """

    def __init__(self, model_id, classes, *args, **kwargs):
        """
        Initialize the BasePredictor.

        Args:
            model_id (str): Identifier for the model to use.
            classes (List[str]): List of possible class labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.model_id = model_id
        self.classes = classes

    @abstractmethod
    def predict(
        self,
        prompts: List[str],
        xs: np.ndarray,
    ) -> np.ndarray:
        """
        Abstract method to make predictions based on prompts and input data.

        Args:
            prompts (List[str]): List of prompts to use for prediction.
            xs (np.ndarray): Array of input data.

        Returns:
            np.ndarray: Array of predictions.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError


class DummyPredictor(BasePredictor):
    """
    A dummy predictor implementation for testing purposes.

    This predictor generates random predictions from the list of possible classes.

    Attributes:
        model_id (str): Always set to "dummy".
        classes (List[str]): List of possible class labels.

    Methods:
        predict: Generates random predictions for the given prompts and input data.
    """

    def __init__(self, model_id, classes, *args, **kwargs):
        self.model_id = "dummy"
        self.classes = classes

    def predict(
        self,
        prompts: List[str],
        xs: np.ndarray,
    ) -> np.ndarray:
        """
        Generate random predictions for the given prompts and input data.

        Args:
            prompts (List[str]): List of prompts (ignored in this implementation).
            xs (np.ndarray): Array of input data (only the length is used).

        Returns:
            np.ndarray: 2D array of random predictions, shape (len(prompts), len(xs)).
        """
        return np.array([np.random.choice(self.classes, len(xs)) for _ in prompts])
