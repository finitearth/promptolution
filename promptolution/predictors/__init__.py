from promptolution.llms import get_llm

from .base_predictor import DummyPredictor
from .classificator import Classificator


def get_predictor(name, *args, **kwargs):
    """
    Factory function to create and return a predictor instance based on the provided name.

    This function supports two types of predictors:
    1. DummyPredictor: A mock predictor for testing purposes.
    2. Classificator: A real predictor using a language model for classification tasks.

    Args:
        name (str): Identifier for the predictor to use. Special case:
                    - "dummy" for DummyPredictor
                    - Any other string for Classificator with the specified LLM
        *args: Variable length argument list passed to the predictor constructor.
        **kwargs: Arbitrary keyword arguments passed to the predictor constructor.

    Returns:
        An instance of DummyPredictor or Classificator based on the name.

    Notes:
        - For non-dummy predictors, this function calls get_llm to obtain the language model.
        - The batch_size for the language model is currently commented out and not used.

    Examples:
        >>> dummy_pred = get_predictor("dummy", classes=["A", "B", "C"])
        >>> real_pred = get_predictor("gpt-3.5-turbo", classes=["positive", "negative"])
    """
    if name == "dummy":
        return DummyPredictor("", *args, **kwargs)

    downstream_llm = get_llm(name)  # , batch_size=config.downstream_bs)

    return Classificator(downstream_llm, *args, **kwargs)
