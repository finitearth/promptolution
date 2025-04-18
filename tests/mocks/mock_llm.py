from typing import List

from promptolution.llms.base_llm import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM for testing purposes.
    
    This class allows precise control over responses for testing without loading actual models.
    """
    
    def __init__(self, predetermined_responses=None, *args, **kwargs):
        """Initialize the MockLLM with optional predetermined responses.
        
        Args:
            predetermined_responses (Dict or List): Mapping from prompts to responses,
                or a list of responses to return in sequence.
            *args, **kwargs: Arguments to pass to BaseLLM.
        """
        super().__init__(*args, **kwargs)
        self.predetermined_responses = predetermined_responses or {}
        self.call_history = []
        self.response_index = 0
    
    def _get_response(self, prompts: List[str], system_prompts: List[str]) -> List[str]:
        """Generate predetermined responses for the given prompts.
        
        Records the inputs for later verification in tests.
        
        Args:
            prompts (List[str]): Input prompts
            system_prompts (List[str]): System prompts
            
        Returns:
            List[str]: Predetermined responses
        """
        # Record the call for test assertions
        self.call_history.append({
            'prompts': prompts,
            'system_prompts': system_prompts
        })
        
        # Handle case where there's a single system prompt for multiple prompts
        if len(system_prompts) == 1 and len(prompts) > 1:
            system_prompts = system_prompts * len(prompts)
        
        results = []
        for i, prompt in enumerate(prompts):
            # Handle dictionary-based responses
            if isinstance(self.predetermined_responses, dict):
                # Try exact match first
                if prompt in self.predetermined_responses:
                    results.append(self.predetermined_responses[prompt])
                # Try system prompt combination
                elif i < len(system_prompts) and (prompt, system_prompts[i]) in self.predetermined_responses:
                    results.append(self.predetermined_responses[(prompt, system_prompts[i])])
                # Default response
                else:
                    results.append(f"Mock response for: {prompt}")
            # Handle list-based responses (return in sequence)
            elif isinstance(self.predetermined_responses, list):
                if self.response_index < len(self.predetermined_responses):
                    results.append(self.predetermined_responses[self.response_index])
                    self.response_index += 1
                else:
                    results.append(f"Mock response for: {prompt}")
            # Default fallback
            else:
                results.append(f"Mock response for: {prompt}")
                
        return results
    
    def reset(self):
        """Reset the call history and response index."""
        self.call_history = []
        self.response_index = 0