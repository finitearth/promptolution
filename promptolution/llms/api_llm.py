import asyncio
import requests
import time
from logging import INFO, Logger

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models.deepinfra import ChatDeepInfraException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from promptolution.llms.deepinfra import ChatDeepInfra


logger = Logger(__name__)
logger.setLevel(INFO)


async def invoke_model(prompt, model, semaphore):
    async with semaphore:
        max_retries = 100
        delay = 3
        attempts = 0

        while attempts < max_retries:
            try:
                response = await asyncio.to_thread(model.invoke, [HumanMessage(content=prompt)])
                return response.content
            except ChatDeepInfraException as e:
                print(f"DeepInfra error: {e}. Attempt {attempts}/{max_retries}. Retrying in {delay} seconds...")
                attempts += 1
                time.sleep(delay)


class APILLM:
    def __init__(self, model_id: str):
        if "claude" in model_id:
            ANTHROPIC_API_KEY = open("anthropictoken.txt", "r").read()
            self.model = ChatAnthropic(model=model_id, api_key=ANTHROPIC_API_KEY)
        elif "gpt" in model_id:
            OPENAI_API_KEY = open("openaitoken.txt", "r").read()
            self.model = ChatOpenAI(model=model_id, api_key=OPENAI_API_KEY)
        elif "llama" in model_id:
            DEEPINFRA_API_KEY = open("deepinfratoken.txt", "r").read()
            self.model = ChatDeepInfra(model_name=model_id, deepinfra_api_token=DEEPINFRA_API_KEY)
        else:
            raise ValueError(f"Unknown model: {model_id}")

    def get_response(self, prompts: list[str]) -> list[str]:
        max_retries = 100
        delay = 3
        attempts = 0

        while attempts < max_retries:
            try:
                responses = asyncio.run(self._get_response(prompts))
                return responses
            except requests.exceptions.ConnectionError as e:
                attempts += 1
                logger.critical(
                    f"Connection error: {e}. Attempt {attempts}/{max_retries}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)

        # If the loop exits, it means max retries were reached
        raise requests.exceptions.ConnectionError("Max retries exceeded. Connection could not be established.")

    async def _get_response(self, prompts: list[str], max_concurrent_calls=200) -> list[str]:
        semaphore = asyncio.Semaphore(max_concurrent_calls)  # Limit the number of concurrent calls
        tasks = []

        for prompt in prompts:
            tasks.append(invoke_model(prompt, self.model, semaphore))

        responses = await asyncio.gather(*tasks)
        return responses
