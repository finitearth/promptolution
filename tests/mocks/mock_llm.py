"""Simplified MockLLM class for testing purposes."""

from typing import List, Optional
from promptolution.llms.base_llm import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM for testing purposes.

    This class allows precise control over responses for testing without loading actual models.
    Simplified to work with list-based responses only.
    """

    def __init__(self, predetermined_responses=None, add_prompt_tags=False, *args, **kwargs):
        """Initialize the MockLLM with optional predetermined responses.

        Args:
            predetermined_responses (List): List of responses to return in sequence.
                If None, will return generic mock responses.
            add_prompt_tags (bool): Whether to wrap responses in <prompt> tags
            *args, **kwargs: Arguments to pass to BaseLLM.
        """
        super().__init__(*args, **kwargs)

        # Set up response list
        if predetermined_responses is None:
            self.responses = []
        else:
            self.responses = list(predetermined_responses)  # Ensure it's a list

        # Add prompt tags if requested
        if add_prompt_tags:
            self.responses = [
                f"<prompt>{r}</prompt>" if not (r.startswith("<prompt>") and r.endswith("</prompt>")) else r
                for r in self.responses
            ]

        self.call_history = []
        self.response_index = 0
        self._generation_seed = None

    def _get_response(self, prompts: List[str], system_prompts: Optional[List[str]] = None) -> List[str]:
        """Generate predetermined responses for the given prompts.

        Records the inputs for later verification in tests.

        Args:
            prompts (List[str]): Input prompts
            system_prompts (Optional[List[str]]): System prompts, defaults to None

        Returns:
            List[str]: Predetermined responses
        """
        # Record the call for test assertions
        self.call_history.append({"prompts": prompts, "system_prompts": system_prompts})

        results = []
        for i, prompt in enumerate(prompts):
            # Return the next response from the list if available
            if self.response_index < len(self.responses):
                results.append(self.responses[self.response_index])
                self.response_index += 1
            else:
                # Default response if we've exhausted the list
                if hasattr(self, "add_prompt_tags") and getattr(self, "add_prompt_tags"):
                    results.append(f"<prompt>Mock response for: {prompt}</prompt>")
                else:
                    results.append(f"Mock response for: {prompt}")

        return results

    def set_generation_seed(self, seed: int) -> None:
        """Set the generation seed (no-op, just for API compatibility).

        Args:
            seed: Random seed value
        """
        self._generation_seed = seed

    def reset(self):
        """Reset the call history and response index."""
        self.call_history = []
        self.response_index = 0
