"""Base module for LLMs in the promptolution library."""

import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from promptolution.templates import DEFAULT_SYS_PROMPT

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for Language Models in the promptolution library.

    This class defines the interface that all concrete LLM implementations should follow.

    Methods:
        get_response: An abstract method that should be implemented by subclasses
                      to generate responses for given prompts.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the LLM."""
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

    def get_response(self, prompts: List[str], system_prompts: List[str] = None) -> List[str]:
        """Generate responses for the given prompts.

        This method calls the _get_response method to generate responses
        for the given prompts. It also updates the token count for the
        input and output tokens.

        Args:
            prompts (str or List[str]): Input prompt(s). If a single string is provided,
                                        it's converted to a list containing that string.
            system_prompts (str or List[str]): System prompt(s) to provide context to the model.

        Returns:
            List[str]: A list of generated responses, one for each input prompt.
        """
        if system_prompts is None:
            system_prompts = DEFAULT_SYS_PROMPT
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        responses = self._get_response(prompts, system_prompts)
        self.update_token_count(prompts + system_prompts, responses)

        return responses

    @abstractmethod
    def _get_response(self, prompts: List[str], system_prompts: List[str] = None) -> List[str]:
        """Generate responses for the given prompts.

        This method should be implemented by subclasses to define how
        the LLM generates responses.

        Args:
            prompts (List[str]): A list of input prompts.
            system_prompts (List[str]): A list of system prompts to provide context to the model.

        Returns:
            List[str]: A list of generated responses corresponding to the input prompts.
        """
        raise NotImplementedError


class DummyLLM(BaseLLM):
    """A dummy implementation of the BaseLLM for testing purposes.

    This class generates random responses for given prompts, simulating
    the behavior of a language model without actually performing any
    complex natural language processing.
    """

    def _get_response(self, prompts: str) -> str:
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
        for p in prompts:
            r = np.random.rand()
            if r < 0.3:
                results += [f"Joooo wazzuppp <prompt>hier gehts los {r} </prompt> {p}"]
            elif 0.3 <= r < 0.6:
                results += [f"was das hier? <prompt>peter lustig{r}</prompt> {p}"]
            else:
                results += [f"hier ist ein <prompt>test{r}</prompt> {p}"]

        return results
