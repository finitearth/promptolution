from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseLLM(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_response(self, prompts: List[str]) -> List[str]:
        pass


class DummyLLM(BaseLLM):
    def __init__(self, *args, **kwargs):
        pass

    def get_response(self, prompts: str) -> str:
        if isinstance(prompts, str):
            prompts = [prompts]
        results = []
        for _ in prompts:
            r = np.random.rand()
            if r < 0.3:
                results += [f"Joooo wazzuppp <prompt>hier gehts los {r} </prompt>"]
            if 0.3 <= r < 0.6:
                results += [f"was das hier? <prompt>peter lustig{r}</prompt>"]
            else:
                results += [f"hier ist ein <prompt>test{r}</prompt>"]

        return results
