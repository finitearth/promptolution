import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatDeepInfra
import asyncio

# possible model names we'll use here are:
# gpt-4o-2024-05-13, gpt-3.5-turbo-0125
# claude-3-opus-20240229, claude-3-haiku-20240307


OPENAI_API_KEY = open("openaitoken.txt", "r").read()
ANTHROPIC_API_KEY = open("anthropictoken.txt", "r").read()
GROQ_API_KEY = open("groqtoken.txt", "r").read()
DEEPINFRA_API_KEY = open("deepinfratoken.txt", "r").read()


        
async def invoke_model(prompt, model, semaphore):
    async with semaphore:
        response = await asyncio.to_thread(model.invoke, [HumanMessage(content=prompt)])
        return response.content


class APILLM:
    def __init__(self, model_id: str):
        if "claude" in model_id:
            self.model = ChatAnthropic(model=model_id, api_key=ANTHROPIC_API_KEY) # TODO check if chat interface or other interface?
        elif "gpt" in model_id:
            # we may use the BatchAPI instead of the LangChain interface here so save some costs
            self.model = ChatOpenAI(model=model_id, api_key=OPENAI_API_KEY)
        elif "llama" in model_id:
            # self.model = ChatGroq(model=model_id, api_key=GROQ_API_KEY)
            self.model = ChatDeepInfra(model_id=model_id, deepinfra_api_token=DEEPINFRA_API_KEY)
        else:
            raise ValueError(f"Unknown model: {model_id}")
    
    def get_response(self, prompts: list[str]) -> list[str]:
        responses = asyncio.run(self._get_response(prompts))
        return responses

    async def _get_response(self, prompts: list[str], max_concurrent_calls=200) -> list[str]:
        semaphore = asyncio.Semaphore(max_concurrent_calls)  # Limit the number of concurrent calls
        tasks = []
        
        for prompt in prompts:
            tasks.append(invoke_model(prompt, self.model, semaphore))
        
        responses = await asyncio.gather(*tasks)
        return responses

