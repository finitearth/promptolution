"""Configuration class for the promptolution library."""
import warnings
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

import pandas as pd

logger = Logger(__name__)


class ExperimentConfig:
    """Configuration class for the promptolution library.

    This is a unified configuration class that handles all experiment settings.
    It provides validation and tracking of used fields.
    """

    def __init__(self, **kwargs):
        """Initialize the configuration with the provided keyword arguments."""
        # Track which attributes are actually used
        self._used_attributes: Set[str] = set()

        # Task settings
        self.task_name: Optional[str] = None
        self.dataset: Optional[Union[Path, pd.DataFrame]] = None
        self.dataset_description: Optional[str] = None
        self.init_prompts: Optional[List[str]] = None
        self.task_description: Optional[str] = None
        self.n_eval_samples: int = 20
        self.n_ds_samples_to_meta: int = 2
        self.prepend_exemplars: bool = False
        self.n_exemplars: int = 5
        self.exemplar_selector: Optional[str] = None
        self.classes: Optional[List[str]] = None

        # LLM settings
        self.meta_llm: Optional[str] = None
        self.downstream_llm: Optional[str] = None
        self.evaluation_llm: Optional[str] = None
        self.meta_bs: Optional[int] = None
        self.downstream_bs: Optional[int] = None
        self.api_token: Optional[str] = None
        self.meta_prompt: Optional[str] = None
        self.model_storage_path: Optional[Path] = Path("../models/")

        # Optimizer settings
        self.optimizer: Optional[str] = None
        self.n_steps: Optional[int] = None
        self.init_pop_size: Optional[int] = None
        self.donor_random: bool = False
        self.selection_mode: Literal["random", "wheel", "tour"] = "random"

        # Predictor settings
        self.predictor: Literal[
            "MarkerBasedClassificator", "FirstOccurenceClassificator"
        ] = "FirstOccurenceClassificator"

        # Set default values from class variables
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("_") and not callable(value):
                setattr(self, key, value)

        # Set attributes from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unexpected configuration parameter '{key}'")

        # Validate after initialization
        self._validate()

    def __getattribute__(self, name):
        """Override attribute access to track used attributes."""
        # Get the attribute using the standard mechanism
        value = object.__getattribute__(self, name)

        if not name.startswith("_") and not callable(value):
            self._used_attributes.add(name)

        return value

    def _validate(self):
        """Validate the configuration settings."""
        # Validate LLM settings
        if self.meta_llm is not None:
            if "local" in self.meta_llm and self.meta_bs is None:
                raise ValueError("'meta_bs' must be specified for local meta_llm")
            if self.downstream_llm and "local" in self.downstream_llm and self.downstream_bs is None:
                raise ValueError("'downstream_bs' must be specified for local downstream_llm")
        if self.api_token is None:
            warnings.warn("No API token provided. Using default tokens from token files.")

    def validate(self):
        """Check if any attributes were not used and run validation.

        Does not raise an error, but logs a warning if any attributes are unused or validation fails.
        """
        all_attributes = {k for k in self.__dict__ if not k.startswith("_")}
        unused_attributes = all_attributes - self._used_attributes
        if unused_attributes:
            logger.warning(f"Unused configuration attributes: {unused_attributes}")

        self._validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns:
            Dict: Configuration as a dictionary.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
