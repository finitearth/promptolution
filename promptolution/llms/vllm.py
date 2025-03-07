"""Module for running language models locally using the vLLM library."""


from logging import INFO, Logger

try:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import vllm, torch or transformers in vllm.py: {e}")

from promptolution.llms.base_llm import BaseLLM

logger = Logger(__name__)
logger.setLevel(INFO)


class VLLM(BaseLLM):
    """A class for running language models using the vLLM library.

    This class sets up a vLLM inference engine with specified model parameters
    and provides a method to generate responses for given prompts.

    Attributes:
        llm (vllm.LLM): The vLLM inference engine.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        sampling_params (vllm.SamplingParams): Parameters for text generation.

    Methods:
        get_response: Generate responses for a list of prompts.
        get_token_count: Get the current count of input and output tokens.
        reset_token_count: Reset the token counters to zero.
    """

    def __init__(
        self,
        model_id: str,
        batch_size: int | None = None,
        max_generated_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        model_storage_path: str | None = None,
        token: str | None = None,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 2048,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """Initialize the VLLM with a specific model.

        Args:
            model_id (str): The identifier of the model to use.
            batch_size (int, optional): The batch size for text generation. Defaults to 8.
            max_generated_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
            model_storage_path (str, optional): Directory to store the model. Defaults to None.
            token: (str, optional): Token for accessing the model - not used in implementation yet.
            dtype (str, optional): Data type for model weights. Defaults to "float16".
            tensor_parallel_size (int, optional): Number of GPUs for tensor parallelism. Defaults to 1.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to use. Defaults to 0.95.
            max_model_len (int, optional): Maximum sequence length for the model. Defaults to 2048.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the LLM class initialization.

        Note:
            This method sets up a vLLM engine with specified parameters for efficient inference.
        """
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code

        # Configure sampling parameters
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_generated_tokens)

        # Initialize the vLLM engine with both explicit parameters and any additional kwargs
        llm_params = {
            "model": model_id,
            "tokenizer": model_id,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "download_dir": model_storage_path,
            "trust_remote_code": self.trust_remote_code,
            **kwargs,
        }

        self.llm = LLM(**llm_params)

        if batch_size is None:
            gpu_blocks = self.llm.llm_engine.model_executor.cache_config.num_gpu_blocks
            block_size = self.llm.llm_engine.model_executor.cache_config.block_size
            self.batch_size = int((gpu_blocks * block_size / self.max_model_len) * 0.95)
            logger.info(f"Batch size set to {self.batch_size} based on GPU memory.")
        else:
            self.batch_size = batch_size

        # Initialize tokenizer separately for potential pre-processing
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _get_response(self, inputs: list[str]):
        """Generate responses for a list of prompts using the vLLM engine.

        Args:
            prompts (list[str]): A list of input prompts.

        Returns:
            list[str]: A list of generated responses corresponding to the input prompts.

        Note:
            This method uses vLLM's batched generation capabilities for efficient inference.
            It also counts input and output tokens.
        """
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": input},
                ],
                tokenize=False,
            )
            for input in inputs
        ]

        # Count input tokens
        for prompt in prompts:
            input_tokens = self.tokenizer.encode(prompt)
            self.input_token_count += len(input_tokens)

        # generate responses for self.batch_size prompts at the same time
        all_responses = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            outputs = self.llm.generate(batch, self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]

            all_responses.extend(responses)

        return all_responses

    def __del__(self):
        """Cleanup method to delete the LLM instance and free up GPU memory."""
        del self.llm
        torch.cuda.empty_cache()
