"""Module for LLM predictors."""

from typing import Literal

from .base_predictor import DummyPredictor
from .classificator import FirstOccurrenceClassificator, MarkerBasedClassificator


def get_predictor(
    downstream_llm=None, type: Literal["first_occurence", "marker"] = "first_occurrence", *args, **kwargs
):
    """Factory function to create and return a predictor instance.

    This function supports three types of predictors:
    1. DummyPredictor: A mock predictor for testing purposes when no downstream_llm is provided.
    2. FirstOccurrenceClassificator: A predictor that classifies based on first occurrence of the label.
    3. MarkerBasedClassificator: A predictor that classifies based on a marker.

    Args:
        downstream_llm: The language model to use for prediction. If None, returns a DummyPredictor.
        type (Literal["first_occurrence", "marker"]): The type of predictor to create:
                    - "first_occurrence" (default) for FirstOccurrenceClassificator
                    - "marker" for MarkerBasedClassificator
        *args: Variable length argument list passed to the predictor constructor.
        **kwargs: Arbitrary keyword arguments passed to the predictor constructor.

    Returns:
        An instance of DummyPredictor, FirstOccurrenceClassificator, or MarkerBasedClassificator.
    """
    if downstream_llm is None:
        return DummyPredictor("", *args, **kwargs)

    if type == "first_occurrence":
        return FirstOccurrenceClassificator(downstream_llm, *args, **kwargs)
    elif type == "marker":
        return MarkerBasedClassificator(downstream_llm, *args, **kwargs)
    else:
        raise ValueError(f"Invalid predictor type: '{type}'")
