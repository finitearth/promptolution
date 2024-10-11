"""Configuration class for the promptolution library."""
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Configuration class for the promptolution library.

    This class handles loading and parsing of configuration settings,
    either from a config file or from keyword arguments.

    Attributes:
        task_name (str): Name of the task.
        ds_path (str): Path to the dataset.
        n_steps (int): Number of optimization steps.
        optimizer (str): Name of the optimizer to use.
        meta_llm (str): Name of the meta language model.
        downstream_llm (str): Name of the downstream language model.
        evaluation_llm (str): Name of the evaluation language model.
        init_pop_size (int): Initial population size. Defaults to 10.
        logging_dir (str): Directory for logging. Defaults to "logs/run.csv".
        experiment_name (str): Name of the experiment. Defaults to "experiment".
        include_task_desc (bool): Whether to include task description. Defaults to False.
        donor_random (bool): Whether to use random donor prompts for EvoPromptDE. Defaults to False.
        random_seed (int): Random seed for reproducibility. Defaults to 42.
        selection_mode (str): Selection mode for EvoPromptGA. Defaults to "random".
        meta_bs (int): Batch size for local meta LLM. Defaults to None.
        downstream_bs (int): Batch size for local downstream LLM. Defaults to None.
        api_token (str): API token for different APIs, implemented as LLM class. Defaults to None.
        meta_prompt (str): Prompt template for the meta LLM. Defaults to None.
    """

    task_name: str = None
    ds_path: Path = None
    optimizer: str = None
    meta_llm: str = None
    downstream_llm: str = None
    evaluation_llm: str = None
    n_steps: int = None
    init_pop_size: int = None
    logging_dir: Path = Path("logs/run.csv")
    experiment_name: str = "experiment"
    include_task_desc: bool = True
    donor_random: bool = False
    random_seed: int = 42
    selection_mode: Optional[str] = "random"
    meta_bs: Optional[int] = None
    downstream_bs: Optional[int] = None
    api_token: Optional[str] = None
    meta_prompt: Optional[str] = None

    def __post_init__(self):
        """Validate the configuration after initialization."""
        self._validate_config()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(**cls._process_config_dict(config_dict))

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
