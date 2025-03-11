"""Base module for predictors."""

from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from promptolution.llms.base_llm import BaseLLM


class BasePredictor:
    """Abstract base class for predictors in the promptolution library.

    This class defines the interface that all concrete predictor implementations should follow.

    Attributes:
        llm: The language model used for generating predictions.


    Methods:
        predict: An abstract method that should be implemented by subclasses
                 to make predictions based on prompts and input data.
    """

    def __init__(self, llm: BaseLLM):
        """Initialize the BasePredictor.

        Args:
            llm: The language model to use for predictions.
            classes (List[str]): The list of valid class labels.
        """
        self.llm = llm

    def predict(self, prompts: List[str], xs: np.ndarray, return_seq: bool = False) -> np.ndarray:
        """Abstract method to make predictions based on prompts and input data.

        Args:
            prompts (List[str]): List of prompts to use for prediction.
            xs (np.ndarray): Array of input data.
            return_seq (bool, optional): whether to return the generating sequence

        Returns:
            np.ndarray: Array of predictions.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        outputs = self.llm.get_response([prompt + "\n" + x for prompt in prompts for x in xs])
        preds = self._extract_preds(outputs)

        shape = (len(prompts), len(xs))
        outputs = np.array(outputs).reshape(shape)
        preds = preds.reshape(shape)
        xs = np.array(xs)

        if return_seq:
            seqs = []
            for o in outputs:
                seqs.append([f"{x}\n{oo}" for x, oo in zip(xs, o)])

            seqs = np.array(seqs)
            # seqs = np.array([''.join(str(i) for i in zip(xs, o)) for o in outputs])

            return preds, seqs

        return preds

    def _extract_preds(self, preds: List[str], shape: Tuple[int, int]) -> np.ndarray:
        """Extract class labels from the predictions, based on the list of valid class labels.

        Args:
            preds: The raw predictions from the language model.
            shape: The shape of the output array: (n_prompts, n_samples).
        """
        raise NotImplementedError


class DummyPredictor(BasePredictor):
    """A dummy predictor implementation for testing purposes.

    This predictor generates random predictions from the list of possible classes.

    Attributes:
        model_id (str): Always set to "dummy".
        classes (List[str]): List of possible class labels.

    Methods:
        predict: Generates random predictions for the given prompts and input data.
    """

    def __init__(self, model_id, classes, *args, **kwargs):
        """Initialize the DummyPredictor.

        Parameters
        ----------
        model_id : str
            Model identifier string.
        classes : list
            List of possible class labels.
        """
        self.model_id = "dummy"
        self.classes = classes

    def predict(
        self,
        prompts: List[str],
        xs: np.ndarray,
    ) -> np.ndarray:
        """Generate random predictions for the given prompts and input data.

        Args:
            prompts (List[str]): List of prompts (ignored in this implementation).
            xs (np.ndarray): Array of input data (only the length is used).

        Returns:
            np.ndarray: 2D array of random predictions, shape (len(prompts), len(xs)).
        """
        return np.array([np.random.choice(self.classes, len(xs)) for _ in prompts])
