from .api_llm import APILLM
from .base_llm import DummyLLM
from .local_llm import LocalLLM


def get_llm(model_id: str, *args, **kwargs):
    """
    Factory function to create and return a language model instance based on the provided model_id.

    This function supports three types of language models:
    1. DummyLLM: A mock LLM for testing purposes.
    2. LocalLLM: For running models locally (identified by 'local' in the model_id).
    3. APILLM: For API-based models (default if not matching other types).

    Args:
        model_id (str): Identifier for the model to use. Special cases:
                        - "dummy" for DummyLLM
                        - "local-{model_name}" for LocalLLM
                        - Any other string for APILLM
        *args: Variable length argument list passed to the LLM constructor.
        **kwargs: Arbitrary keyword arguments passed to the LLM constructor.

    Returns:
        An instance of DummyLLM, LocalLLM, or APILLM based on the model_id.
    """
    if model_id == "dummy":
        return DummyLLM(*args, **kwargs)
    if "local" in model_id:
        model_id = "-".join(model_id.split("-")[1:])
        return LocalLLM(model_id, *args, **kwargs)
    return APILLM(model_id, *args, **kwargs)
