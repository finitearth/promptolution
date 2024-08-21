
try:
    import torch
    import transformers
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import torch or transformers in local_llm.py: {e}")

class LocalLLM:
    def __init__(self):
        self.model = torch.nn.Module()  # Safe to use torch since it's conditionally imported


class LocalLLM:
    def __init__(self, model_id: str, batch_size=8):

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

    def get_response(self, prompts: list[str]):
        with torch.no_grad():
            response = self.pipeline(prompts, pad_token_id=self.pipeline.tokenizer.eos_token_id)

        if len(response) != 1:
            response = [r[0] if isinstance(r, list) else r for r in response]

        response = [r["generated_text"] for r in response]
        return response
    
