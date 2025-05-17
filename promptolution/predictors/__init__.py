"""Module for LLM predictors."""

from typing import Literal

from .base_predictor import DummyPredictor
from .classifier import FirstOccurrenceClassifier, MarkerBasedClassifier


def get_predictor(downstream_llm=None, type: Literal["first_occurrence", "marker"] = "marker", *args, **kwargs):
    """Factory function to create and return a predictor instance.

    This function supports three types of predictors:
    1. DummyPredictor: A mock predictor for testing purposes when no downstream_llm is provided.
    2. FirstOccurrenceClassifier: A predictor that classifies based on first occurrence of the label.
    3. MarkerBasedClassifier: A predictor that classifies based on a marker.

    Args:
        downstream_llm: The language model to use for prediction. If None, returns a DummyPredictor.
        type (Literal["first_occurrence", "marker"]): The type of predictor to create:
                    - "first_occurrence" (default) for FirstOccurrenceClassifier
                    - "marker" for MarkerBasedClassifier
        *args: Variable length argument list passed to the predictor constructor.
        **kwargs: Arbitrary keyword arguments passed to the predictor constructor.

    Returns:
        An instance of DummyPredictor, FirstOccurrenceClassifier, or MarkerBasedClassifier.
    """
    if downstream_llm is None:
        return DummyPredictor("", *args, **kwargs)

    if type == "first_occurrence":
        return FirstOccurrenceClassifier(downstream_llm, *args, **kwargs)
    elif type == "marker":
        return MarkerBasedClassifier(downstream_llm, *args, **kwargs)
    else:
        raise ValueError(f"Invalid predictor type: '{type}'")
