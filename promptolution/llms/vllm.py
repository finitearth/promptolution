"""Module for running language models locally using the vLLM library."""


from logging import INFO, Logger

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = Logger(__name__)
logger.setLevel(INFO)


class VLLM:
    """A class for running language models using the vLLM library.

    This class sets up a vLLM inference engine with specified model parameters
    and provides a method to generate responses for given prompts.

    Attributes:
        llm (vllm.LLM): The vLLM inference engine.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        sampling_params (vllm.SamplingParams): Parameters for text generation.

    Methods:
        get_response: Generate responses for a list of prompts.
    """

    def __init__(
        self,
        model_id: str,
        batch_size: int = 8,
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        model_storage_path: str = None,
        token: str = None,
    ):
        """Initialize the VLLM with a specific model.

        Args:
            model_id (str): The identifier of the model to use.
            batch_size (int, optional): The batch size for text generation. Defaults to 8.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
            model_storage_path (str, optional): Directory to store the model. Defaults to None.
            token: (str, optional): Token for accessing the model - not used in implementation yet.

        Note:
            This method sets up a vLLM engine with specified parameters for efficient inference.
        """
        # Configure sampling parameters
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        # Initialize the vLLM engine
        self.llm = LLM(
            model=model_id,
            tokenizer=model_id,
            dtype="float16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2048,
            download_dir=model_storage_path,
            trust_remote_code=True,
        )

        # Initialize tokenizer separately for potential pre-processing
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.batch_size = batch_size

    def get_response(self, inputs: list[str]):
        """Generate responses for a list of prompts using the vLLM engine.

        Args:
            prompts (list[str]): A list of input prompts.

        Returns:
            list[str]: A list of generated responses corresponding to the input prompts.

        Note:
            This method uses vLLM's batched generation capabilities for efficient inference.
        """
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful, harmless, and honest assistant. "
                        "You answer the user's questions accurately and fairly.",
                    },
                    {"role": "user", "content": input},
                ],
                tokenize=False,
            )
            for input in inputs
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)
        responses = [output.outputs[0].text for output in outputs]

        return responses

    def __del__(self):
        """Cleanup method to delete the LLM instance and free up GPU memory."""
        del self.llm
        torch.cuda.empty_cache()
