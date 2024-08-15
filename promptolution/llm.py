import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


OPENAI_API_KEY = open("openaitoken.txt", "r").read()
ANTHROPIC_API_KEY = open("anthropictoken.txt", "r").read()

# possible model names we'll use here are:
# gpt-4o-2024-05-13, gpt-3.5-turbo-0125
# claude-3-opus-20240229, claude-3-haiku-20240307

def get_llm(model_id: str, *args, **kwargs):
    if model_id == "dummy":
        return DummyLLM(*args, **kwargs)
    if "gpt" or "claude" in model_id:
        return APILLM(model_id, *args, **kwargs)
    else:
        return LocalLLM(model_id, *args, **kwargs)


class APILLM: # TODO function to read from config and initialize predictor
    def __init__(self, model_id: str):
        if "claude" in model_id:
            self.model = ChatAnthropic(model=model_id, api_key=ANTHROPIC_API_KEY) # TODO check if chat interface or other interface?
        elif "gpt" in model_id:
            # we may use the BatchAPI instead of the LangChain interface here so save some costs
            self.model = ChatOpenAI(model=model_id, api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown model: {model_id}")

    def get_response(self, prompt: str):
        response = self.model.invoke([HumanMessage(content=prompt)]).content
        return response


class LocalLLM:
    def __init__(self, model_id: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # set pad token to eos
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def get_response(self, prompt: str):
        x = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**x, max_length=1024, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class DummyLLM:
    def __init__(self, *args, **kwargs):
        pass
    def get_response(self, prompt: str) -> str:
        r = np.random.rand()
        if r < 0.3:
            return f"Joooo wazzuppp <prompt>hier gehts los {r} </prompt>"
        if 0.3 <= r < 0.6:
            return f"was das hier? <prompt>peter lustig{r}</prompt>"
        return f"hier ist ein <prompt>test{r}</prompt>"
