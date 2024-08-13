import numpy as np

from promptolution.llm import LLM


class Predictor:
    def __init__(self, model_id: str):
        self.llm = LLM(model_id)

    def predict(
        self,
        prompt: str,
        xs: np.ndarray,
    ) -> np.ndarray:
        response = []
        for x in xs:
            response += self.llm.get_response(prompt + str(x))  
        return np.array([response])


class DummyPredictor(Predictor):
    def __init__(self, model_id):
        self.model_id = "dummy"

    def predict(
        self,
        prompt: str,
        xs: np.ndarray,
    ) -> np.ndarray:
        return np.array(["Dummy Answer"])