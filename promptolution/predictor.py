import numpy as np

from promptolution.llm import get_llm

def get_predictor(model_id: str):
    if model_id == "dummy":
        return DummyPredictor(model_id)
    
    return Predictor(model_id)


class Predictor: # TODO function for initializing predictor from config
    def __init__(self, model_id: str):
        self.llm = get_llm(model_id)

    def predict(
        self,
        prompt: str,
        xs: np.ndarray,
        classes: list[str] = ["Sports", "Tech", "Business", "World"] # TODO change this to be more general and read from config
    ) -> np.ndarray:
        response = []
        for x in xs:
            pred = self.llm.get_response(prompt + "\n" + str(x))
            predicted_class = ""
            for word in pred.split(" "):
                word = word.replace(".", "").replace(",", "").replace("!", "").replace("?", "")# remove punctuation
                if word in classes:
                    predicted_class = word
                    break

            response.append(predicted_class)
            

        return np.array([response])


class DummyPredictor(Predictor):
    def __init__(self, model_id):
        self.model_id = "dummy"

    def predict(
        self,
        prompt: str,
        xs: np.ndarray,
    ) -> np.ndarray:
        return np.array(["Entity"])
