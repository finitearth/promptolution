"""Configuration class for the promptolution library."""
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Configuration class for the promptolution library."""

    task_name: str
    ds_path: Path
    n_steps: int
    optimizer: str
    meta_llm: str
    downstream_llm: str
    evaluation_llm: str
    init_pop_size: int = None
    logging_dir: Path = Path("logs/run.csv")
    experiment_name: str = "experiment"
    include_task_desc: bool = True
    donor_random: bool = False
    random_seed: int = 42
    selection_mode: Optional[str] = None
    meta_bs: Optional[int] = None
    downstream_bs: Optional[int] = None
    api_token: Optional[str] = None

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
        if self.optimizer == "evopromptga" and not self.selection_mode:
            raise ValueError("'selection_mode' must be specified for 'evopromptga' optimizer")
        if self.optimizer == "evopromptde" and self.donor_random is None:
            raise ValueError("'donor_random' must be specified for 'evopromptde' optimizer")
        if "local" in self.meta_llm and self.meta_bs is None:
            raise ValueError("'meta_bs' must be specified for local meta_llm")
        if "local" in self.downstream_llm and self.downstream_bs is None:
            raise ValueError("'downstream_bs' must be specified for local downstream_llm")
        if self.api_token is None:
            print("Warning: No API token provided. Using default tokens from token files.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Config instance to a dictionary."""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
