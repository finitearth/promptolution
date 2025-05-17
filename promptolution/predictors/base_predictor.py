"""Base module for predictors in the promptolution library."""

from abc import ABC, abstractmethod
from typing import List, Optional

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
        self,
        prompts: List[str],
        xs: np.ndarray,
        system_prompts: List[str] = None,
        return_seq: bool = False,
    ) -> np.ndarray:
        """Abstract method to make predictions based on prompts and input data.

        Args:
            prompts: Prompt or list of prompts to use for prediction.
            xs: Array of input data.
            system_prompts: List of system prompts to use for the language model.
            return_seq: Whether to return the generating sequence.

        Returns:
            Array of predictions, optionally with sequences.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        inputs = [prompt + "\n" + x for prompt, x in zip(prompts, xs)]
        outputs = self.llm.get_response(inputs, system_prompts=system_prompts)
        preds = self._extract_preds(outputs)

        if return_seq:
            seqs = [f"{x}\n{out}" for x, out in zip(xs, outputs)]
            seqs = np.array(seqs)

        return preds if not return_seq else (preds, seqs)

    @abstractmethod
    def _extract_preds(self, preds: List[str]) -> np.ndarray:
        """Extract class labels from the predictions, based on the list of valid class labels.

        Args:
            preds: The raw predictions from the language model.

        Returns:
            np.ndarray: Extracted predictions.
        """
        raise NotImplementedError
