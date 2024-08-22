from typing import List
import numpy as np
from promptolution.predictors.base_predictor import BasePredictor

class Classificator(BasePredictor): 
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