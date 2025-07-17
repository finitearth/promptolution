"""Module to interface with various language models through their respective APIs."""


try:
    import asyncio

    from openai import AsyncOpenAI

    import_successful = True
except ImportError:
    import_successful = False


from typing import TYPE_CHECKING, List

from promptolution.llms.base_llm import BaseLLM

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.config import ExperimentConfig

from promptolution.utils.logging import get_logger

logger = get_logger(__name__)


async def _invoke_model(prompt, system_prompt, max_tokens, model_id, client, semaphore, max_retries=20, retry_delay=5):
    async with semaphore:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        for attempt in range(max_retries + 1):  # +1 for the initial attempt
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return response
            except Exception as e:
                if attempt < max_retries:
                    # Calculate exponential backoff with jitter
                    logger.warning(
                        f"⚠️ API call failed (attempt {attempt + 1} / {max_retries + 1}): {str(e)}. "
                        f"Retrying in {retry_delay:.2f} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    # Log the final failure and re-raise the exception
                    logger.error(f"❌ API call failed after {max_retries + 1} attempts: {str(e)}")
                    raise


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
        self,
        api_url: str = None,
        model_id: str = None,
        api_key: str = None,
        max_concurrent_calls=50,
        max_tokens=512,
        config: "ExperimentConfig" = None,
    ):
        """Initialize the APILLM with a specific model and API configuration.

        Args:
            api_url (str): The base URL for the API endpoint.
            model_id (str): Identifier for the model to use.
            api_key (str, optional): API key for authentication. Defaults to None.
            max_concurrent_calls (int, optional): Maximum number of concurrent API calls. Defaults to 50.
            max_tokens (int, optional): Maximum number of tokens in model responses. Defaults to 512.
            config (ExperimentConfig, optional): Configuration for the LLM, overriding defaults.

        Raises:
            ImportError: If required libraries are not installed.
        """
        if not import_successful:
            raise ImportError(
                "Could not import at least one of the required libraries: openai, asyncio. "
                "Please ensure they are installed in your environment."
            )

        self.api_url = api_url
        self.model_id = model_id
        self.api_key = api_key
        self.max_concurrent_calls = max_concurrent_calls
        self.max_tokens = max_tokens

        super().__init__(config=config)
        self.client = AsyncOpenAI(base_url=self.api_url, api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(self.max_concurrent_calls)

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
