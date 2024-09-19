import asyncio
import requests
import time
import openai
from logging import INFO, Logger

from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models.deepinfra import ChatDeepInfraException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from promptolution.llms.deepinfra import ChatDeepInfra


logger = Logger(__name__)
logger.setLevel(INFO)


async def invoke_model(prompt, model, semaphore):
    """
    Asynchronously invoke a language model with retry logic.

    Args:
        prompt (str): The input prompt for the model.
        model: The language model to invoke.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent calls.

    Returns:
        str: The model's response content.

    Raises:
        ChatDeepInfraException: If all retry attempts fail.
    """
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
    """
    A class to interface with various language models through their respective APIs.

    This class supports Claude (Anthropic), GPT (OpenAI), and LLaMA (DeepInfra) models.
    It handles API key management, model initialization, and provides methods for
    both synchronous and asynchronous inference.

    Attributes:
        model: The initialized language model instance.

    Methods:
        get_response: Synchronously get responses for a list of prompts.
        _get_response: Asynchronously get responses for a list of prompts.
    """
    def __init__(self, model_id: str):
        """
        Initialize the APILLM with a specific model.

        Args:
            model_id (str): Identifier for the model to use.

        Raises:
            ValueError: If an unknown model identifier is provided.
        """
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

    def get_response(self, prompts: List[str]) -> List[str]:
        """
        Synchronously get responses for a list of prompts.

        This method includes retry logic for handling connection errors and rate limits.

        Args:
            prompts (list[str]): List of input prompts.

        Returns:
            list[str]: List of model responses.

        Raises:
            requests.exceptions.ConnectionError: If max retries are exceeded.
        """
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
            except openai.RateLimitError as e:
                attempts += 1
                logger.critical(
                    f"Rate limit error: {e}. Attempt {attempts}/{max_retries}. Retrying in {delay} seconds..."
                )
                time.sleep(delay)

        # If the loop exits, it means max retries were reached
        raise requests.exceptions.ConnectionError("Max retries exceeded. Connection could not be established.")

    async def _get_response(
        self, prompts: list[str], max_concurrent_calls=200
    ) -> list[str]:  # TODO change name of method
        """
        Asynchronously get responses for a list of prompts.

        This method uses a semaphore to limit the number of concurrent API calls.

        Args:
            prompts (list[str]): List of input prompts.
            max_concurrent_calls (int): Maximum number of concurrent API calls allowed.

        Returns:
            list[str]: List of model responses.
        """
        semaphore = asyncio.Semaphore(max_concurrent_calls)  # Limit the number of concurrent calls
        tasks = []

        for prompt in prompts:
            tasks.append(invoke_model(prompt, self.model, semaphore))

        responses = await asyncio.gather(*tasks)
        return responses
