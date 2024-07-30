import numpy as np

class Task:
    def __init__(self, predictor):
        self.predictor = predictor

    def evaluate(self, prompt):
        pass


class ClassificationTask(Task):
    def __init__(self, predictor, xs: np.ndarray, ys: np.ndarray):
        super().__init__(predictor)
        self.xs = xs
        self.ys = ys

    def evaluate(self, prompt):
        preds = self.predictor.predict(prompt, self.xs)
        return np.mean(preds == self.ys)
