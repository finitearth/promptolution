"""Token counter for LLMs.

This module provides a function to count the number of tokens in a given text.
"""

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM
    from transformers import PreTrainedTokenizer
from promptolution.utils.logging import get_logger

logger = get_logger(__name__)


def get_token_counter(llm: "BaseLLM") -> Callable[[str], int]:
    """Get a token counter function for the given LLM.

    This function returns a callable that counts tokens based on the LLM's tokenizer
    or a simple split method if no tokenizer is available.

    Args:
        llm: The language model object that may have a tokenizer.

    Returns:
        A callable that takes a text input and returns the token count.

    """
    if llm.tokenizer is not None:
        tokenizer: PreTrainedTokenizer = llm.tokenizer
        return lambda x: len(tokenizer.encode(x))
    else:
        logger.warning("⚠️ The LLM does not have a tokenizer. Using simple token count.")
        return lambda x: len(x.split())
