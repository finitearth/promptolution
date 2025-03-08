"""Module for LLM predictors."""

from promptolution.llms import get_llm

from .base_predictor import DummyPredictor
from .classificator import FirstOccurrenceClassificator, MarkerBasedClassificator


def get_predictor(downstream_llm=None, type: str = "first_occurrence", *args, **kwargs):
    """Factory function to create and return a predictor instance based on the provided name.

    This function supports two types of predictors:
    1. DummyPredictor: A mock predictor for testing purposes.
    2. FirstOccurrenceClassificator: A real predictor using a language model for classification tasks.

    Args:
        name (str): Identifier for the predictor to use. Special case:
                    - "dummy" for DummyPredictor
                    - Any other string for FirstOccurrenceClassificator with the specified LLM
        type ()
        *args: Variable length argument list passed to the predictor constructor.
        **kwargs: Arbitrary keyword arguments passed to the predictor constructor.

    Returns:
        An instance of DummyPredictor or FirstOccurrenceClassificator based on the name.

    Notes:
        - For non-dummy predictors, this function calls get_llm to obtain the language model.
        - The batch_size for the language model is currently commented out and not used.

    Examples:
        >>> dummy_pred = get_predictor("dummy", classes=["A", "B", "C"])
        >>> real_pred = get_predictor("gpt-3.5-turbo", classes=["positive", "negative"])
    """
    if downstream_llm is None:
        return DummyPredictor("", *args, **kwargs)

    if type == "first_occurrence":
        return FirstOccurrenceClassificator(downstream_llm, *args, **kwargs)
    elif type == "marker":
        return MarkerBasedClassificator(downstream_llm, *args, **kwargs)
