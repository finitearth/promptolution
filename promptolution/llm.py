import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatDeepInfra
import asyncio
# import torch
# import transformers


OPENAI_API_KEY = open("openaitoken.txt", "r").read()
ANTHROPIC_API_KEY = open("anthropictoken.txt", "r").read()
GROQ_API_KEY = open("groqtoken.txt", "r").read()
DEEPINFRA_API_KEY = open("deepinfratoken.txt", "r").read()

# possible model names we'll use here are:
# gpt-4o-2024-05-13, gpt-3.5-turbo-0125
# claude-3-opus-20240229, claude-3-haiku-20240307

def get_llm(model_id: str, *args, **kwargs):
    if model_id == "dummy":
        return DummyLLM(*args, **kwargs)
    # if "local" in model_id:
    #     model_id = "-".join(model_id.split("-")[1:])
    #     return LocalLLM(model_id, *args, **kwargs)
    return APILLM(model_id, *args, **kwargs)


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
        

async def invoke_model(prompt, model, semaphore):
    async with semaphore:
        response = await asyncio.to_thread(model.invoke, [HumanMessage(content=prompt)])
        return response.content


    # def get_response(self, prompts: list[str]) -> list[str]:
    #     responses = []
    #     for prompt in prompts:
    #         response = self.model.invoke([HumanMessage(content=prompt)]).content
    #         responses.append(response)
    #         print("oh oh!")
    #         print(response)
    #     return responses

# class LocalLLM:
#     def __init__(self, model_id: str, batch_size=8):
#         self.pipeline = transformers.pipeline(
#             "text-generation", 
#             model=model_id,
#             model_kwargs={"torch_dtype": torch.bfloat16},
#             device_map="auto", 
#             max_new_tokens=256,
#             batch_size=batch_size,
#             num_return_sequences=1,
#             return_full_text=False,
#         )
#         self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id

#     @torch.no_grad()
#     def get_response(self, prompts: list[str]):
#         response = self.pipeline(prompts, pad_token_id=self.pipeline.tokenizer.eos_token_id)

#         if len(response) != 1:
#             response = [r[0] if isinstance(r, list) else r for r in response]

#         response = [r["generated_text"] for r in response]
#         return response


class DummyLLM:
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
