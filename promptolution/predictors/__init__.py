from .classificator import Classificator
from .base_predictor import DummyPredictor
from promptolution.llms import get_llm

def get_predictor(config, *args, **kwargs):
    if config.downstream_llm == "dummy":
        return DummyPredictor("", *args, **kwargs)
    
    downstream_llm = get_llm(config.downstream_llm)#, batch_size=config.downstream_bs)
    
    return Classificator(downstream_llm, *args, **kwargs)