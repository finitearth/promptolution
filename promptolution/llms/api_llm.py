"""Module to interface with various language models through their respective APIs."""


try:
    import asyncio

    from openai import AsyncOpenAI

    import_successful = True
except ImportError:
    import_successful = False

from logging import Logger
from typing import Any, List

from promptolution.llms.base_llm import BaseLLM

logger = Logger(__name__)


async def _invoke_model(prompt, system_prompt, max_tokens, model_id, client, semaphore):
    async with semaphore:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response


class APILLM(BaseLLM):
    """A class to interface with language models through their respective APIs.

    This class provides a unified interface for making API calls to language models
    using the OpenAI client library. It handles rate limiting through semaphores
    and supports both synchronous and asynchronous operations.

    Attributes:
        model_id (str): Identifier for the model to use.
        client (AsyncOpenAI): The initialized API client.
        max_tokens (int): Maximum number of tokens in model responses.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent API calls.
    """

    def __init__(
        self, api_url: str, model_id: str, token: str = None, max_concurrent_calls=50, max_tokens=512, **kwargs: Any
    ):
        """Initialize the APILLM with a specific model and API configuration.

        Args:
            api_url (str): The base URL for the API endpoint.
            model_id (str): Identifier for the model to use.
            token (str, optional): API key for authentication. Defaults to None.
            max_concurrent_calls (int, optional): Maximum number of concurrent API calls. Defaults to 50.
            max_tokens (int, optional): Maximum number of tokens in model responses. Defaults to 512.
            **kwargs (Any): Additional parameters to pass to the API client.

        Raises:
            ImportError: If required libraries are not installed.
        """
        if not import_successful:
            raise ImportError(
                "Could not import at least one of the required libraries: openai, asyncio. "
                "Please ensure they are installed in your environment."
            )
        super().__init__()
        self.model_id = model_id
        self.client = AsyncOpenAI(base_url=api_url, api_key=token, **kwargs)
        self.max_tokens = max_tokens

        self.semaphore = asyncio.Semaphore(max_concurrent_calls)

    def _get_response(self, prompts: List[str], system_prompts: List[str]) -> List[str]:
        # Setup for async execution in sync context
        loop = asyncio.get_event_loop()
        responses = loop.run_until_complete(self._get_response_async(prompts, system_prompts))
        return responses

    async def _get_response_async(self, prompts: List[str], system_prompts: List[str]) -> List[str]:
        tasks = [
            _invoke_model(prompt, system_prompt, self.max_tokens, self.model_id, self.client, self.semaphore)
            for prompt, system_prompt in zip(prompts, system_prompts)
        ]
        responses = await asyncio.gather(*tasks)
        return [response.choices[0].message.content for response in responses]
