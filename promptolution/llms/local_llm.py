"""Module for running LLMs locally using the Hugging Face Transformers library."""
try:
    import torch
    import transformers

    imports_successful = True
except ImportError:
    imports_successful = False

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.config import ExperimentConfig


from promptolution.llms.base_llm import BaseLLM


class LocalLLM(BaseLLM):
    """A class for running language models locally using the Hugging Face Transformers library.

    This class sets up a text generation pipeline with specified model parameters
    and provides a method to generate responses for given prompts.

    Attributes:
        pipeline (transformers.Pipeline): The text generation pipeline.

    Methods:
        get_response: Generate responses for a list of prompts.
    """

    def __init__(self, model_id: str, batch_size: int = 8, config: "ExperimentConfig" = None):
        """Initialize the LocalLLM with a specific model.

        Args:
            model_id (str): The identifier of the model to use (e.g., "gpt2", "facebook/opt-1.3b").
            batch_size (int, optional): The batch size for text generation. Defaults to 8.
            config (ExperimentConfig, optional): "ExperimentConfig" overwriting defaults.

        Note:
            This method sets up a text generation pipeline with bfloat16 precision,
            automatic device mapping, and specific generation parameters.
        """
        if not imports_successful:
            raise ImportError(
                "Could not import at least one of the required libraries: torch, transformers. "
                "Please ensure they are installed in your environment."
            )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_new_tokens=256,
            batch_size=batch_size,
            num_return_sequences=1,
            return_full_text=False,
        )
        self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
        self.pipeline.tokenizer.padding_side = "left"
        super().__init__(config)

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        """Generate responses for a list of prompts using the local language model.

        Args:
            prompts (list[str]): A list of input prompts.
            system_prompts (list[str]): A list of system prompts to guide the model's behavior.

        Returns:
            list[str]: A list of generated responses corresponding to the input prompts.

        Note:
            This method uses torch.no_grad() for inference to reduce memory usage.
            It handles both single and batch inputs, ensuring consistent output format.
        """
        inputs = []
        for prompt, sys_prompt in zip(prompts, system_prompts):
            inputs.append([{"role": "system", "prompt": sys_prompt}, {"role": "user", "prompt": prompt}])

        with torch.no_grad():
            response = self.pipeline(inputs, pad_token_id=self.pipeline.tokenizer.eos_token_id)

        if len(response) != 1:
            response = [r[0] if isinstance(r, list) else r for r in response]

        response = [r["generated_text"] for r in response]
        return response

    def __del__(self):
        """Cleanup method to delete the pipeline and free up GPU memory."""
        del self.pipeline
        torch.cuda.empty_cache()
