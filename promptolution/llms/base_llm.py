"""Base module for LLMs in the promptolution library."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseLLM(ABC):
    """Abstract base class for Language Models in the promptolution library.

    This class defines the interface that all concrete LLM implementations should follow.

    Methods:
        get_response: An abstract method that should be implemented by subclasses
                      to generate responses for given prompts.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the LLM."""
        pass

    @abstractmethod
    def get_response(self, prompts: List[str]) -> List[str]:
        """Generate responses for the given prompts.

        This method should be implemented by subclasses to define how
        the LLM generates responses.

        Args:
            prompts (List[str]): A list of input prompts.

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
