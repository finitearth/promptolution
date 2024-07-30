
class Predictor:
    def __init__(self):
        pass

    def predict(self, prompt, text):
        pass

class APIPredictor(Predictor):
    def __init__(self, model):
        if "claude" in model:
            from langchain_anthropic import ChatAnthropic
            self.model = ChatAnthropic(model)
        elif "gpt" in model:
            from langchain_openai import ChatOpenAI
            self.model = ChatOpenAI(model)
        else:
            raise ValueError(f"Unknown model: {model}")
            
    def predict(self, prompt, text):
        return self.model.predict(prompt, text)