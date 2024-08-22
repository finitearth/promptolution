from .api_llm import APILLM
from .base_llm import DummyLLM
from .local_llm import LocalLLM


def get_llm(model_id: str, *args, **kwargs):
    if model_id == "dummy":
        return DummyLLM(*args, **kwargs)
    if "local" in model_id:
        model_id = "-".join(model_id.split("-")[1:])
        return LocalLLM(model_id, *args, **kwargs)
    return APILLM(model_id, *args, **kwargs)
