from typing import Optional
from langchain_anthropic import AnthropicLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import numpy as np

from promptolution.tasks import Task

OPENAI_API_KEY = open("openaitoken.txt", "r").read()
ANTHROPIC_API_KEY = open("anthropictoken.txt", "r").read()


class Predictor:
    def __init__(self, model_id: str, task: Task):
        if "claude" in model_id:
            self.model = AnthropicLLM(model=model_id, api_key=ANTHROPIC_API_KEY)
        elif "gpt" in model_id:
            self.model = ChatOpenAI(model=model_id, api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown model: {model_id}")

    def predict(
        self,
        prompt: Optional[str],
        xs: np.ndarray,
    ):
        chain = PromptTemplate.from_template(prompt) | self.model
        return chain.invoke({"question": str(xs)})


class DummyPredictor(Predictor):
    def __init__(self, model_id):
        self.model_id = "dummy"

    def predict(
        self,
        prompt: str,
        xs: np.ndarray,
    ) -> np.ndarray:
        return np.array(["Dummy Answer"])
