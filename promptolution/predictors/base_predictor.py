"""Base module for predictors in the promptolution library."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from promptolution.config import ExperimentConfig
from promptolution.llms.base_llm import BaseLLM


class BasePredictor(ABC):
    """Abstract base class for predictors in the promptolution library.

    This class defines the interface that all concrete predictor implementations should follow.

    Attributes:
        llm: The language model used for generating predictions.
        classes (List[str]): The list of valid class labels.
        config (ExperimentConfig): Experiment configuration overwriting defaults
    """

    def __init__(self, llm: Optional[BaseLLM] = None, config: ExperimentConfig = None):
        """Initialize the predictor with a language model and configuration.

        Args:
            llm: Language model to use for prediction.
            config: Configuration for the predictor.
        """
        self.llm = llm

        if config is not None:
            config.apply_to(self)

    def predict(
        self, prompts: List[str], xs: np.ndarray, system_prompts: List[str] = None, return_seq: bool = False
    ) -> np.ndarray:
        """Abstract method to make predictions based on prompts and input data.

        Args:
            prompts: Prompt or list of prompts to use for prediction.
            xs: Array of input data.
            return_seq: Whether to return the generating sequence.

        Returns:
            Array of predictions, optionally with sequences.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        outputs = self.llm.get_response(
            [prompt + "\n" + x for prompt in prompts for x in xs], system_prompts=system_prompts
        )
        preds = self._extract_preds(outputs)

        shape = (len(prompts), len(xs))
        outputs = np.array(outputs).reshape(shape)
        preds = preds.reshape(shape)
        xs = np.array(xs)

        if return_seq:
            seqs = []
            for output in outputs:
                seqs.append([f"{x}\n{out}" for x, out in zip(xs, output)])

            seqs = np.array(seqs)

            return preds, seqs

        return preds

    @abstractmethod
    def _extract_preds(self, preds: List[str], shape: Tuple[int, int]) -> np.ndarray:
        """Extract class labels from the predictions, based on the list of valid class labels.

        Args:
            preds: The raw predictions from the language model.
            shape: The shape of the output array: (n_prompts, n_samples).

        Returns:
            np.ndarray: Extracted predictions with shape (n_prompts, n_samples).
        """
        raise NotImplementedError


class DummyPredictor(BasePredictor):
    """A dummy predictor implementation for testing purposes.

    This predictor generates random predictions from the list of possible classes.

    Attributes:
        model_id (str): Always set to "dummy".
        classes (List[str]): List of possible class labels.
    """

    def predict(
        self, prompts: Union[str, List[str]], xs: np.ndarray, return_seq: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """Generate random predictions for the given prompts and input data.

        Args:
            prompts: Prompt or list of prompts (ignored in this implementation).
            xs: Array of input data (only the length is used).
            return_seq: Whether to return sequences.

        Returns:
            Array of random predictions, optionally with sequences.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        preds = np.array([np.random.choice(self.classes, len(xs)) for _ in prompts])

        if return_seq:
            # Generate fake sequences
            seqs = [f"Input: {x}\nOutput: {np.random.choice(self.classes)}" for x in xs]
            return preds, seqs

        return preds

    def _extract_preds(self, preds: List[str], shape: Tuple[int, int]) -> np.ndarray:
        """Extract class labels from the predictions.

        This is a dummy implementation that returns random predictions.

        Args:
            preds: The raw predictions from the language model (ignored).
            shape: The shape of the output array: (n_prompts, n_samples).

        Returns:
            np.ndarray: Random predictions.
        """
        return np.array([np.random.choice(self.classes, shape[1]) for _ in range(shape[0])])
