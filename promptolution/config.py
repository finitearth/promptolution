"""Configuration classes for the promptolution library."""
import configparser
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

import pandas as pd


class BaseConfig:
    """Base configuration class for all components.

    This class provides validation and tracking of used fields.
    """

    def __init__(self, **kwargs):
        """Initialize the configuration with the provided keyword arguments."""
        # Track which attributes are actually used
        self._used_attributes: Set[str] = set()

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

    def _validate(self):
        """Validate the configuration.

        This method should be implemented by subclasses to enforce specific validation rules.
        """
        pass

    def mark_as_used(self, attribute_name: str):
        """Mark an attribute as used.

        Args:
            attribute_name: Name of the attribute that was used.
        """
        if hasattr(self, attribute_name):
            self._used_attributes.add(attribute_name)

    def check_unused_attributes(self):
        """Check if any attributes were not used.

        Returns a list of warnings for unused attributes.
        """
        all_attributes = {k for k in self.__dict__ if not k.startswith("_")}
        unused_attributes = all_attributes - self._used_attributes
        return [f"Warning: Config attribute '{attr}' was not used" for attr in unused_attributes]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns:
            Dict: Configuration as a dictionary.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class TaskConfig:
    """Configuration for task settings."""

    task_name: Optional[str] = None
    dataset: Optional[Union[Path, pd.DataFrame]] = None
    dataset_description: Optional[str] = None
    init_prompts: Optional[List[str]] = None
    task_description: Optional[str] = None
    n_eval_samples: int = 20
    n_ds_samples_to_meta: int = 2
    prepend_exemplars: bool = False
    n_exemplars: int = 5
    exemplar_selector: Optional[str] = None
    classes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LLMConfig:
    """Configuration for language models."""

    meta_llm: Optional[str] = None
    downstream_llm: Optional[str] = None
    evaluation_llm: Optional[str] = None
    meta_bs: Optional[int] = None
    downstream_bs: Optional[int] = None
    api_token: Optional[str] = None
    meta_prompt: Optional[str] = None
    model_storage_path: Optional[Path] = Path("../models/")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""

    optimizer: Optional[str] = None
    n_steps: Optional[int] = None
    init_pop_size: Optional[int] = None
    donor_random: bool = False
    selection_mode: Literal["random", "wheel", "tour"] = "random"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""

    experiment_name: str = "experiment"
    logging_dir: Path = Path("logs/run.csv")
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PromptolutionConfig(BaseConfig):
    """Configuration class for the promptolution library.

    This config class combines all component configs and follows the Hugging Face style
    of tracking which configuration attributes are actually used.
    """

    def __init__(
        self,
        task_config: Optional[Union[Dict[str, Any], TaskConfig]] = None,
        llm_config: Optional[Union[Dict[str, Any], LLMConfig]] = None,
        optimizer_config: Optional[Union[Dict[str, Any], OptimizerConfig]] = None,
        experiment_config: Optional[Union[Dict[str, Any], ExperimentConfig]] = None,
        predictor: Literal["MarkerBasedClassificator", "FirstOccurenceClassificator"] = "FirstOccurenceClassificator",
        **kwargs,
    ):
        """Initialize the configuration with component configs."""
        super().__init__(**kwargs)

        # Initialize component configs
        self.task_config = (
            TaskConfig(**(task_config or {})) if isinstance(task_config, dict) else task_config or TaskConfig()
        )
        self.llm_config = LLMConfig(**(llm_config or {})) if isinstance(llm_config, dict) else llm_config or LLMConfig()
        self.optimizer_config = (
            OptimizerConfig(**(optimizer_config or {}))
            if isinstance(optimizer_config, dict)
            else optimizer_config or OptimizerConfig()
        )
        self.experiment_config = (
            ExperimentConfig(**(experiment_config or {}))
            if isinstance(experiment_config, dict)
            else experiment_config or ExperimentConfig()
        )
        self.predictor = predictor

    def _validate(self):
        """Validate the configuration settings."""
        # Validate LLM settings
        if self.llm_config.meta_llm is not None:
            if "local" in self.llm_config.meta_llm and self.llm_config.meta_bs is None:
                raise ValueError("'meta_bs' must be specified for local meta_llm")
            if (
                self.llm_config.downstream_llm
                and "local" in self.llm_config.downstream_llm
                and self.llm_config.downstream_bs is None
            ):
                raise ValueError("'downstream_bs' must be specified for local downstream_llm")
        if self.llm_config.api_token is None:
            warnings.warn("No API token provided. Using default tokens from token files.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "predictor": self.predictor,
            "task_config": self.task_config.to_dict() if self.task_config else {},
            "llm_config": self.llm_config.to_dict() if self.llm_config else {},
            "optimizer_config": self.optimizer_config.to_dict() if self.optimizer_config else {},
            "experiment_config": self.experiment_config.to_dict() if self.experiment_config else {},
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PromptolutionConfig":
        """Create a configuration instance from a dictionary."""
        task_config = config_dict.pop("task_config", {})
        llm_config = config_dict.pop("llm_config", {})
        optimizer_config = config_dict.pop("optimizer_config", {})
        experiment_config = config_dict.pop("experiment_config", {})

        return cls(
            task_config=task_config,
            llm_config=llm_config,
            optimizer_config=optimizer_config,
            experiment_config=experiment_config,
            **config_dict,
        )

    @classmethod
    def from_file(cls, config_path: Path) -> "PromptolutionConfig":
        """Create a Config instance from a configuration file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Parse as INI file
        config = configparser.ConfigParser()
        config.read(config_path)

        # Initialize empty dicts for each component
        task_config = {}
        llm_config = {}
        optimizer_config = {}
        experiment_config = {}
        main_config = {}

        # Map sections to component configs
        section_mapping = {
            "task": task_config,
            "llm": llm_config,
            "optimizer": optimizer_config,
            "experiment": experiment_config,
            "main": main_config,
        }

        # Parse each section based on the mapping
        for section in config.sections():
            target_dict = section_mapping.get(section.lower(), main_config)
            for key, value in config[section].items():
                target_dict[key] = value

        # Create the config instance
        return cls(
            task_config=task_config,
            llm_config=llm_config,
            optimizer_config=optimizer_config,
            experiment_config=experiment_config,
            **main_config,
        )

    # Legacy support for old Config interface
    @classmethod
    def legacy_from_dict(cls, config_dict: Dict[str, Any]) -> "PromptolutionConfig":
        """Create a Config instance from a flat dictionary (legacy support)."""
        # Initialize component configs
        task_config = {}
        llm_config = {}
        optimizer_config = {}
        experiment_config = {}
        main_config = {}

        # Map old flat keys to component configs
        task_keys = [
            "task_name",
            "dataset",
            "dataset_description",
            "init_prompts",
            "task_description",
            "n_eval_samples",
            "n_ds_samples_to_meta",
            "prepend_exemplars",
            "n_exemplars",
            "exemplar_selector",
        ]

        llm_keys = [
            "meta_llm",
            "downstream_llm",
            "evaluation_llm",
            "meta_bs",
            "downstream_bs",
            "api_token",
            "meta_prompt",
            "model_storage_path",
        ]

        optimizer_keys = ["optimizer", "n_steps", "init_pop_size", "donor_random", "selection_mode"]

        experiment_keys = ["experiment_name", "logging_dir", "random_seed"]

        # Sort keys into respective dicts
        for key, value in config_dict.items():
            if key in task_keys:
                task_config[key] = value
            elif key in llm_keys:
                llm_config[key] = value
            elif key in optimizer_keys:
                optimizer_config[key] = value
            elif key in experiment_keys:
                experiment_config[key] = value
            else:
                main_config[key] = value

        # Create the config instance
        return cls(
            task_config=task_config,
            llm_config=llm_config,
            optimizer_config=optimizer_config,
            experiment_config=experiment_config,
            **main_config,
        )

    # # For backward compatibility
    # @property
    # def task_name(self):
    #     self.mark_as_used("task_config")
    #     return self.task_config.task_name

    # @property
    # def dataset(self):
    #     self.mark_as_used("task_config")
    #     return self.task_config.dataset

    # @property
    # def n_steps(self):
    #     self.mark_as_used("optimizer_config")
    #     return self.optimizer_config.n_steps

    # @property
    # def optimizer(self):
    #     self.mark_as_used("optimizer_config")
    #     return self.optimizer_config.optimizer

    # @property
    # def meta_llm(self):
    #     self.mark_as_used("llm_config")
    #     return self.llm_config.meta_llm

    # @property
    # def downstream_llm(self):
    #     self.mark_as_used("llm_config")
    #     return self.llm_config.downstream_llm

    # @property
    # def evaluation_llm(self):
    #     self.mark_as_used("llm_config")
    #     return self.llm_config.evaluation_llm

    # # Add more backward compatibility properties as needed
    # @property
    # def task_description(self):
    #     self.mark_as_used("task_config")
    #     return self.task_config.task_description

    # @property
    # def donor_random(self):
    #     self.mark_as_used("optimizer_config")
    #     return self.optimizer_config.donor_random

    # @property
    # def selection_mode(self):
    #     self.mark_as_used("optimizer_config")
    #     return self.optimizer_config.selection_mode

    # @property
    # def meta_prompt(self):
    #     self.mark_as_used("llm_config")
    #     return self.llm_config.meta_prompt
