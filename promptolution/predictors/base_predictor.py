from abc import abstractmethod
from typing import List

import numpy as np


class BasePredictor:
    def __init__(self, model_id, classes, *args, **kwargs):
        self.model_id = model_id
        self.classes = classes

    @abstractmethod
    def predict(
        self,
        prompts: List[str],
        xs: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class DummyPredictor(BasePredictor):
    def __init__(self, model_id, classes, *args, **kwargs):
        self.model_id = "dummy"
        self.classes = classes

    def predict(
        self,
        prompts: List[str],
        xs: np.ndarray,
    ) -> np.ndarray:
        return np.array([np.random.choice(self.classes, len(xs)) for _ in prompts])
