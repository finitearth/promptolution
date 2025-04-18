# tests/test_base_llm.py
import pytest

from promptolution.llms.base_llm import BaseLLM, DummyLLM
from tests.mocks.mock_llm import MockLLM


def test_base_llm_token_counting():
    """Test token counting functionality."""
    llm = DummyLLM()
    
    # Get initial token count
    initial_count = llm.get_token_count()
    assert initial_count["input_tokens"] == 0
    assert initial_count["output_tokens"] == 0
    
    # Process some text
    prompts = ["This is a test prompt with several words."]
    llm.get_response(prompts)
    
    # Check updated token count
    updated_count = llm.get_token_count()
    assert updated_count["input_tokens"] > 0
    assert updated_count["output_tokens"] > 0
    
    # Reset token count
    llm.reset_token_count()
    reset_count = llm.get_token_count()
    assert reset_count["input_tokens"] == 0
    assert reset_count["output_tokens"] == 0

