import numpy as np
from typing import List
from promptolution.llm import get_llm

def get_predictor(config, *args, **kwargs):
    if config.downstream_llm == "dummy":
        return DummyPredictor("", *args, **kwargs)
    
    downstream_llm = get_llm(config.downstream_llm)#, batch_size=config.downstream_bs)
    
    return Predictor(downstream_llm, *args, **kwargs)


class Predictor: 
    def __init__(self, llm, classes, *args, **kwargs):
        self.llm = llm
        self.classes = classes

    def predict(
        self,
        prompts: List[str],
        xs: np.ndarray,
    ) -> np.ndarray:
        if isinstance(prompts, str):
            prompts = [prompts]
        
        preds = self.llm.get_response([prompt + "\n" + x for prompt in prompts for x in xs])
        
        response = []
        for pred in preds:
            predicted_class = ""
            for word in pred.split(" "):
                word = "".join([c for c in word if c.isalpha()])
                if word in self.classes:
                    predicted_class = word
                    break

            response.append(predicted_class)

        response = np.array(response).reshape(len(prompts), len(xs))
        return response


class DummyPredictor(Predictor):
    def __init__(self, model_id, classes, *args, **kwargs):
        self.model_id = "dummy"
        self.classes = classes

    def predict(
        self,
        prompts: List[str],
        xs: np.ndarray,
    ) -> np.ndarray:
        return np.array([np.random.choice(self.classes, len(xs)) for _ in prompts])