from .classificator import Classificator
from .base_predictor import DummyPredictor
from promptolution.llms import get_llm

def get_predictor(name, *args, **kwargs):
    if name == "dummy":
        return DummyPredictor("", *args, **kwargs)
    
    downstream_llm = get_llm(name)#, batch_size=config.downstream_bs)
    
    return Classificator(downstream_llm, *args, **kwargs)