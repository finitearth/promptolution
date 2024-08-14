import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = open("openaitoken.txt", "r").read()
ANTHROPIC_API_KEY = open("anthropictoken.txt", "r").read()

# possible model names we'll use here are:
# gpt-4o-2024-05-13, gpt-3.5-turbo-0125
# claude-3-opus-20240229, claude-3-haiku-20240307


class LLM:
    def __init__(self, model_id: str):
        if "claude" in model_id:
            self.model = ChatAnthropic(model=model_id, api_key=ANTHROPIC_API_KEY)
        elif "gpt" in model_id:
            # we may use the BatchAPI instead of the LangChain interface here so save some costs
            self.model = ChatOpenAI(model=model_id, api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown model: {model_id}")

    def get_response(self, prompt: str):
        return self.model.invoke([HumanMessage(content=prompt)]).content


class DummyLLM(LLM):
    def get_response(self, prompt: str) -> str:
        r = np.random.rand()
        if r < 0.3:
            return f"Joooo wazzuppp <prompt>hier gehts los {r} </prompt>"
        if 0.3 <= r < 0.6:
            return f"was das hier? <prompt>peter lustig{r}</prompt>"
        return f"hier ist ein <prompt>test{r}</prompt>"
