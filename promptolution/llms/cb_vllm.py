"""Module for running language models using vLLM with continuous batching."""

import time
from concurrent.futures import ThreadPoolExecutor
from logging import INFO, Logger
from queue import Queue
from threading import Lock
from typing import List

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


class ContinuousBatchVLLM(BaseLLM):
    """A class for running language models using vLLM with continuous batching."""

    def __init__(
        self,
        model_id: str,
        concurrent_requests: int = 8,
        max_generated_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        model_storage_path: str = None,
        token: str = None,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 2048,
        trust_remote_code: bool = False,
        block_size: int = 16,
    ):
        """Initialize the continuous batching vLLM engine.

        Args:
            model_id (str): The identifier of the model to use.
            concurrent_requests (int, optional): Number of requests to process concurrently. Defaults to 8.
            max_generated_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
            model_storage_path (str, optional): Directory to store the model. Defaults to None.
            token (str, optional): Token for accessing the model. Defaults to None.
            dtype (str, optional): Data type for model weights. Defaults to "auto".
            tensor_parallel_size (int, optional): Number of GPUs for tensor parallelism. Defaults to 1.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to use. Defaults to 0.95.
            max_model_len (int, optional): Maximum sequence length for the model. Defaults to 2048.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            block_size (int, optional): KV cache block size. Smaller values can improve performance. Defaults to 16.
        """
        self.model_id = model_id
        self.concurrent_requests = concurrent_requests
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.block_size = block_size

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_generated_tokens,
            early_stopping=True,
        )

        logger.info(f"Initializing continuous batching vLLM with model {model_id}")
        start_time = time.time()

        self.llm = LLM(
            model=model_id,
            tokenizer=model_id,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            download_dir=model_storage_path,
            trust_remote_code=self.trust_remote_code,
            block_size=self.block_size,
        )

        logger.info(f"vLLM initialization took {time.time() - start_time:.2f} seconds")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.request_queue = Queue()
        self.result_map = {}
        self.result_lock = Lock()

        self._warm_up_model()

        self.is_running = True
        self.executor.submit(self._continuous_batch_worker)

    def _warm_up_model(self):
        logger.info("Warming up model...")
        start_time = time.time()

        warmup_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            tokenize=False,
        )

        self.llm.generate([warmup_prompt], self.sampling_params)

        torch.cuda.empty_cache()
        logger.info(f"Model warm-up completed in {time.time() - start_time:.2f} seconds")

    def _continuous_batch_worker(self):
        logger.info("Starting continuous batching worker thread")

        active_requests = {}

        while self.is_running:
            while not self.request_queue.empty() and len(active_requests) < self.concurrent_requests:
                try:
                    request_id, prompt = self.request_queue.get_nowait()
                    active_requests[request_id] = prompt
                except Exception:
                    break

            if active_requests:
                try:
                    request_ids = list(active_requests.keys())
                    prompts = list(active_requests.values())

                    logger.info(f"Processing batch of {len(prompts)} prompts")
                    start_time = time.time()

                    outputs = self.llm.generate(prompts, self.sampling_params)

                    elapsed = time.time() - start_time
                    logger.info(f"Batch processed in {elapsed:.3f}s ({len(prompts)/elapsed:.1f} prompts/sec)")

                    with self.result_lock:
                        for request_id, output in zip(request_ids, outputs):
                            self.result_map[request_id] = output.outputs[0].text

                    active_requests.clear()

                except Exception as e:
                    logger.error(f"Error in continuous batching worker: {e}")
                    active_requests.clear()

            time.sleep(0.01)

    def get_response(self, inputs: List[str]) -> List[str]:
        """Generate responses for a list of prompts using the continuous batching vLLM engine.

        This method queues the input prompts for processing by the background worker thread
        and waits for the results to be available.

        Args:
            inputs (List[str]): A list of input prompts.

        Returns:
            List[str]: A list of generated responses corresponding to the input prompts.
        """
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful, harmless, and honest assistant. "
                        "You answer the user's questions accurately and fairly.",
                    },
                    {"role": "user", "content": input_text},
                ],
                tokenize=False,
            )
            for input_text in inputs
        ]

        request_ids = [f"req_{int(time.time() * 1000)}_{i}" for i in range(len(prompts))]

        for request_id, prompt in zip(request_ids, prompts):
            self.request_queue.put((request_id, prompt))

        max_wait_time = 60
        start_time = time.time()

        results = [None] * len(request_ids)
        remaining_ids = set(request_ids)

        while remaining_ids and (time.time() - start_time) < max_wait_time:
            with self.result_lock:
                for i, request_id in enumerate(request_ids):
                    if request_id in self.result_map and request_id in remaining_ids:
                        results[i] = self.result_map[request_id]
                        remaining_ids.remove(request_id)
                        del self.result_map[request_id]

            if remaining_ids:
                time.sleep(0.1)

        if remaining_ids:
            logger.warning(f"Timed out waiting for {len(remaining_ids)} requests")
            for i, request_id in enumerate(request_ids):
                if results[i] is None:
                    results[i] = "Error: Request timed out"

        return results

    def __del__(self):
        """Cleanup method to stop the worker thread and free resources.

        This magic method is called when the object is about to be destroyed.
        It ensures proper shutdown of the background worker thread and
        releases GPU resources.
        """
        self.is_running = False

        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

        if hasattr(self, "llm"):
            del self.llm

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
