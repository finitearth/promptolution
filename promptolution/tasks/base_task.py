from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseTask(ABC):
    def __init__(self, *args, **kwargs):
        pass	

    @abstractmethod
    def evaluate(self, prompts: List[str], predictor) -> np.ndarray:
        raise NotImplementedError
    

class DummyTask(BaseTask):
    def __init__(self):
        self.task_id = "dummy"
        self.dataset_json = None
        self.initial_population = ["Some", "initial", "prompts", "that", "will", "do", "the", "trick"]
        self.description = "This is a dummy task for testing purposes."
        self.xs = np.array(["This is a test", "This is another test", "This is a third test"])
        self.ys = np.array(["positive", "negative", "positive"])
        self.classes = ["negative", "positive"]

    def evaluate(self, prompts: List[str], predictor) -> np.ndarray:
        return np.array([np.random.rand()]*len(prompts))