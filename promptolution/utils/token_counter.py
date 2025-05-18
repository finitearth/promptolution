"""Token counter for LLMs.

This module provides a function to count the number of tokens in a given text.
"""
from promptolution import get_logger

logger = get_logger(__name__)


def get_token_counter(llm):
    """Get a token counter function for the given LLM.

    This function returns a callable that counts tokens based on the LLM's tokenizer
    or a simple split method if no tokenizer is available.

    Args:
        llm: The language model object that may have a tokenizer.

    Returns:
        A callable that takes a text input and returns the token count.

    """
    if hasattr(llm, "tokenizer"):
        token_counter = lambda x: len(llm.tokenizer(x)["input_ids"])
    else:
        logger.warning("⚠️ The LLM does not have a tokenizer. Using simple token count.")
        token_counter = lambda x: len(x.split())

    return token_counter
