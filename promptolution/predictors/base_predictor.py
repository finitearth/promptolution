"""Base module for predictors in the promptolution library."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from promptolution.llms.base_llm import BaseLLM


@dataclass
class PredictorConfig:
    """Configuration for predictor settings.

    This class defines the configuration parameters for predictors.

    Attributes:
        predictor_name (str): Name of the predictor.
        predictor_type (str): Type of prediction task (e.g., classification).
        classes (List[str]): List of class labels for classification tasks.
    """

    predictor_name: str = ""
    predictor_type: str = "classification"
    classes: List[str] = field(default_factory=list)


class BasePredictor(ABC):
    """Abstract base class for predictors in the promptolution library.

    This class defines the interface that all concrete predictor implementations should follow.
    It's designed to follow the Hugging Face-style interface pattern while maintaining
    compatibility with the existing API.

    Attributes:
        config (PredictorConfig): Configuration for the predictor.
        llm: The language model used for generating predictions.
    """

    config_class = PredictorConfig

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        classes: Optional[List[str]] = None,
        config: Optional[Union[Dict[str, Any], PredictorConfig]] = None,
        **kwargs,
    ):
        """Initialize the predictor with a language model and configuration.

        Args:
            llm: Language model to use for prediction.
            classes: List of valid class labels (for backward compatibility).
            config: Configuration for the predictor.
            **kwargs: Additional keyword arguments for configuration.
        """
        # Initialize config
        if config is None:
            config = {}

            # For backward compatibility, add classes to config
            if classes is not None:
                config["classes"] = classes

        if isinstance(config, dict):
            # Merge kwargs into config
            for k, v in kwargs.items():
                config[k] = v
            self.config = self.config_class(**config)
        else:
            self.config = config
            # Override config with kwargs
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        # For backward compatibility
        if hasattr(self.config, "classes") and not hasattr(self, "classes"):
            self.classes = self.config.classes

        # Store LLM for predictions
        self.llm = llm

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

    def __init__(self, model_id=None, classes=None, llm=None, config=None, **kwargs):
        """Initialize the DummyPredictor.

        Args:
            model_id: Model identifier string (ignored, always set to "dummy").
            classes: List of possible class labels.
            llm: Language model (ignored in this implementation).
            config: Configuration for the predictor.
            **kwargs: Additional keyword arguments.
        """
        # Handle both new and old-style initialization
        if config is None and classes is not None:
            config = {"classes": classes}

        super().__init__(llm=llm, config=config, **kwargs)
        self.model_id = "dummy"

        # Ensure classes are available
        if not hasattr(self, "classes") or not self.classes:
            self.classes = ["positive", "negative"]

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
