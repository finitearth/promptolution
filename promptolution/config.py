"""Configuration class for the promptolution library."""
<<<<<<< HEAD
import configparser
=======

from configparser import ConfigParser
>>>>>>> main
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional


@dataclass
class Config:
    """Configuration class for the promptolution library.

    This class handles loading and parsing of configuration settings,
    either from a config file or from keyword arguments.

    Attributes:
        task_name (str): Name of the task. Should not be None if used.
        ds_path (str): Path to the dataset. Should not be None if used.
        n_steps (int): Number of optimization steps. Should not be None if used.
        optimizer (str): Name of the optimizer to use. Should not be None if used.
        meta_llm (str): Name of the meta language model. Should not be None if used.
        downstream_llm (str): Name of the downstream language model. Should not be None if used.
        evaluation_llm (str): Name of the evaluation language model. Should not be None if used.
        init_pop_size (int): Initial population size. Defaults to 10.
        logging_dir (str): Directory for logging. Defaults to "logs/run.csv".
        experiment_name (str): Name of the experiment. Defaults to "experiment".
        include_task_desc (bool): Whether to include task description. Defaults to False.
        donor_random (bool): Whether to use random donor prompts for EvoPromptDE. Defaults to False.
        random_seed (int): Random seed for reproducibility. Defaults to 42.
        selection_mode (str): Selection mode for EvoPromptGA. Defaults to "random".
        meta_bs (int): Batch size for local meta LLM. Should not be None if llm is run locally. Defaults to None.
        downstream_bs (int): Batch size for local downstream LLM.
        Should not be None if llm is run locally Defaults to None.
        api_token (str): API token for different APIs, as implemented in LLM classes.
        Should not be None if APILLM is used. Defaults to None.
        meta_prompt (str): Prompt template for the meta LLM.
        If None is set, default meta_prompts from template.py will be used. Defaults to None.
        prepend_exemplars (bool): rather to do exemplar search and prepend few-shot examples. Defaults to False.
        n_exemplars (int): how many exemplars to prepend. Only used if prepend_exemplars is True. Defaults to 5.
        exemplar_selector (str): which exemplar selector to use. Should not be None if preped_exemplars is True.
        Defaults to None.
        n_ds_samples_to_meta (int): how many examples to show of the ds to show to meta-llm
        (not applicable to every optimizer)
        n_eval_samples (int): how many examples to show to evaluation llm for evaluation.
    """

<<<<<<< HEAD
    task_name: str = None
    ds_path: Path = None
    optimizer: str = None
    meta_llm: str = None
    downstream_llm: str = None
    evaluation_llm: str = None
    n_steps: int = None
    init_pop_size: int = None
    logging_dir: Path = Path("logs/run.csv")
=======
    task_name: str
    ds_path: str
    n_steps: int
    optimizer: str
    meta_prompt_path: str
    meta_llms: str
    downstream_llm: str
    evaluation_llm: str
    init_pop_size: int = 10
    logging_dir: str = "logs/run.csv"
>>>>>>> main
    experiment_name: str = "experiment"
    include_task_desc: bool = True
    donor_random: bool = False
    random_seed: int = 42
    selection_mode: Optional[Literal["random", "wheel", "tour"]] = "random"
    meta_bs: Optional[int] = None
    downstream_bs: Optional[int] = None
    api_token: Optional[str] = None
    meta_prompt: Optional[str] = None
    prepend_exemplars: Optional[bool] = False
    n_exemplars: Optional[int] = 5
    exemplar_selector: Optional[str] = None
    n_ds_samples_to_meta: Optional[int] = 2
    n_eval_samples: Optional[int] = 20

<<<<<<< HEAD
    def __post_init__(self):
        """Validate the configuration after initialization."""
        self._validate_config()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(**cls._process_config_dict(config_dict))
=======
    def __init__(self, config_path: str = None, **kwargs):
        """Initialize the Config object."""
        if config_path:
            self.config_path = config_path
            self.config = ConfigParser()
            self.config.read(config_path)
            self._parse_config()
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _parse_config(self):
        """Parse the configuration settings from the config file."""
        self.task_name = self.config["task"]["task_name"]
        self.ds_path = self.config["task"]["ds_path"]
        self.n_steps = int(self.config["task"]["steps"])
        self.random_seed = int(self.config["task"]["random_seed"])
        self.optimizer = self.config["optimizer"]["name"]
        self.meta_prompt_path = self.config["optimizer"]["meta_prompt_path"]
        self.meta_llm = self.config["meta_llm"]["name"]
        self.downstream_llm = self.config["downstream_llm"]["name"]
        self.evaluation_llm = self.config["evaluator_llm"]["name"]
        self.init_pop_size = int(self.config["optimizer"]["init_pop_size"])
        self.logging_dir = self.config["logging"]["dir"]
        self.experiment_name = self.config["experiment"]["name"]
>>>>>>> main

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Create a Config instance from a configuration file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)

        config_dict = {key: value for section in config.sections() for key, value in config[section].items()}

        return cls.from_dict(config_dict)

    @classmethod
    def _process_config_dict(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate the configuration dictionary."""
        processed_dict = {}
        for field in cls.__dataclass_fields__.values():
            if field.name in config_dict:
                value = config_dict[field.name]
                if field.type == Path:
                    processed_dict[field.name] = Path(value)
                elif field.type == bool:
                    processed_dict[field.name] = str(value).lower() == "true"
                elif field.type == int:
                    processed_dict[field.name] = int(value)
                else:
                    processed_dict[field.name] = value
            elif field.default == field.default_factory:  # Check if field is required
                raise ValueError(f"Required configuration parameter '{field.name}' is missing")

        unknown_args = set(config_dict.keys()) - set(cls.__dataclass_fields__.keys())
        if unknown_args:
            print(f"Warning: Unexpected configuration arguments: {', '.join(unknown_args)}")

        return processed_dict

    def _validate_config(self):
        """Validate the configuration settings."""
        if self.meta_llm is not None:
            if "local" in self.meta_llm and self.meta_bs is None:
                raise ValueError("'meta_bs' must be specified for local meta_llm")
            if "local" in self.downstream_llm and self.downstream_bs is None:
                raise ValueError("'downstream_bs' must be specified for local downstream_llm")
        if self.api_token is None:
            print("Warning: No API token provided. Using default tokens from token files.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Config instance to a dictionary."""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
