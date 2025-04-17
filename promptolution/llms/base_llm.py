"""Base module for LLMs in the promptolution library."""

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from promptolution.config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMModelConfig:
    """Configuration class for language models.

    This class defines the configuration parameters for language models.

    Attributes:
        model_name_or_path (str): The name or path of the model.
        api_base (Optional[str]): The base URL for API requests.
        api_token (Optional[str]): The API token for authentication.
        model_kwargs (Dict[str, Any]): Additional keyword arguments for model initialization.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        temperature (float): The sampling temperature.
        top_p (float): The nucleus sampling probability.
        batch_size (Optional[int]): Batch size for processing requests.
    """

    model_name_or_path: str = ""
    api_base: Optional[str] = None
    api_token: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: Optional[int] = None


class BaseLLM(ABC):
    """Abstract base class for Language Models in the promptolution library.

    This class defines the interface that all concrete LLM implementations should follow.
    It's designed to track which configuration parameters are actually used.

    Attributes:
        config (LLMModelConfig): Configuration for the language model.
        input_token_count (int): Count of input tokens processed.
        output_token_count (int): Count of output tokens generated.
    """

    config_class = LLMModelConfig

    def __init__(self, *args, **kwargs):
        """Initialize the LLM with a configuration or direct parameters.

        This constructor supports both config-based and direct parameter initialization
        for backward compatibility.

        Args:
            *args: Positional arguments (for backward compatibility).
            **kwargs: Keyword arguments either for direct parameters or config fields.
        """
        # Get configuration, either directly or from kwargs
        config = kwargs.pop("config", None)

        # Initialize config
        if config is None:
            # Check if first positional arg is a config
            if args and isinstance(args[0], self.config_class):
                self.config = args[0]
            else:
                # Create config from kwargs
                self.config = self.config_class(**kwargs)
        elif isinstance(config, dict):
            # Create config from dict
            self.config = self.config_class(**config)
        else:
            # Use provided config object
            self.config = config

        # Initialize token counters
        self.input_token_count = 0
        self.output_token_count = 0

    def get_token_count(self):
        """Get the current count of input and output tokens.

        Returns:
            dict: A dictionary containing the input and output token counts.
        """
        return {
            "input_tokens": self.input_token_count,
            "output_tokens": self.output_token_count,
            "total_tokens": self.input_token_count + self.output_token_count,
        }

    def reset_token_count(self):
        """Reset the token counters to zero."""
        self.input_token_count = 0
        self.output_token_count = 0

    def update_token_count(self, inputs: List[str], outputs: List[str]):
        """Update the token count based on the given inputs and outputs.

        Args:
            inputs (List[str]): A list of input prompts.
            outputs (List[str]): A list of generated responses.
        """
        logger.warning("Token count is approximated using word count split by whitespace, not an actual tokenizer.")
        input_tokens = sum([len(i.split()) for i in inputs])
        output_tokens = sum([len(o.split()) for o in outputs])
        self.input_token_count += input_tokens
        self.output_token_count += output_tokens

    def get_response(self, prompts: Union[str, List[str]]) -> List[str]:
        """Generate responses for the given prompts.

        This method calls the _get_response method to generate responses
        for the given prompts. It also updates the token count for the
        input and output tokens.

        Args:
            prompts: Input prompt(s). If a single string is provided,
                    it's converted to a list containing that string.

        Returns:
            List[str]: A list of generated responses, one for each input prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        responses = self._get_response(prompts)
        self.update_token_count(prompts, responses)

        return responses

    @abstractmethod
    def _get_response(self, prompts: List[str]) -> List[str]:
        """Generate responses for the given prompts.

        This method should be implemented by subclasses to define how
        the LLM generates responses.

        Args:
            prompts: A list of input prompts.

        Returns:
            List[str]: A list of generated responses corresponding to the input prompts.
        """
        pass


class DummyLLM(BaseLLM):
    """A dummy implementation of the BaseLLM for testing purposes.

    This class generates random responses for given prompts, simulating
    the behavior of a language model without actually performing any
    complex natural language processing.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the DummyLLM."""
        pass

    def get_response(self, prompts: str) -> str:
        """Generate random responses for the given prompts.

        This method creates silly, random responses enclosed in <prompt> tags.
        It's designed for testing and demonstration purposes.

        Args:
            prompts (str or List[str]): Input prompt(s). If a single string is provided,
                                        it's converted to a list containing that string.

        Returns:
            List[str]: A list of randomly generated responses, one for each input prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        results = []
        for _ in prompts:
            r = np.random.rand()
            if r < 0.3:
                results += [f"Joooo wazzuppp <prompt>hier gehts los {r} </prompt>"]
            if 0.3 <= r < 0.6:
                results += [f"was das hier? <prompt>peter lustig{r}</prompt>"]
            else:
                results += [f"hier ist ein <prompt>test{r}</prompt>"]

        return results
