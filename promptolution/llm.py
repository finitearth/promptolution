import numpy as np

class LLM:
    def predict(self, prompt: str) -> str:
        pass

class DummyLLM(LLM):
    def predict(self, prompt: str) -> str:
        r = np.random.rand()
        if r < 0.3:
            return "Joooo wazzuppp <prompt>hier gehts los </prompt>"
        if 0.3 <= r < 0.6:
            return "was das hier? <prompt>peter lustig</prompt>"
        return "hier ist ein <prompt>test</prompt>"