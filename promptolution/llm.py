import numpy as np

class LLM:
    def predict(self, prompt: str) -> str:
        pass

class DummyLLM(LLM):
    def predict(self, prompt: str) -> str:
        r = np.random.rand()
        if r < 0.3:
            return f"Joooo wazzuppp <prompt>hier gehts los {r} </prompt>"
        if 0.3 <= r < 0.6:
            return f"was das hier? <prompt>peter lustig{r}</prompt>"
        return f"hier ist ein <prompt>test{r}</prompt>"