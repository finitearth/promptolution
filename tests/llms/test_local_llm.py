import pytest
from unittest.mock import MagicMock, patch

from promptolution.llms.local_llm import LocalLLM


@pytest.fixture
def mock_local_dependencies():
    """Set up mocks for LocalLLM dependencies."""
    with patch('promptolution.llms.local_llm.transformers') as mock_transformers, \
         patch('promptolution.llms.local_llm.torch') as mock_torch:
        
        # Configure mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Mock response 1"}, {"generated_text": "Mock response 2"}]
        mock_transformers.pipeline.return_value = mock_pipeline
        
        # Configure mock tokenizer
        mock_pipeline.tokenizer = MagicMock()
        mock_pipeline.tokenizer.pad_token_id = None
        mock_pipeline.tokenizer.eos_token_id = 50256
        mock_pipeline.tokenizer.padding_side = None
        
        yield {
            'transformers': mock_transformers,
            'pipeline': mock_pipeline,
            'torch': mock_torch
        }


def test_local_llm_initialization(mock_local_dependencies):
    """Test that LocalLLM initializes correctly."""
    # Create LocalLLM instance
    local_llm = LocalLLM(
        model_id="gpt2",
        batch_size=4
    )
    
    # Verify pipeline was created correctly
    mock_local_dependencies['transformers'].pipeline.assert_called_once_with(
        "text-generation",
        model="gpt2",
        model_kwargs={"torch_dtype": mock_local_dependencies['torch'].bfloat16},
        device_map="auto",
        max_new_tokens=256,
        batch_size=4,
        num_return_sequences=1,
        return_full_text=False,
    )
    
    # Verify tokenizer attributes were set
    assert local_llm.pipeline.tokenizer.pad_token_id == local_llm.pipeline.tokenizer.eos_token_id
    assert local_llm.pipeline.tokenizer.padding_side == "left"


def test_local_llm_get_response(mock_local_dependencies):
    """Test that LocalLLM._get_response works correctly."""
    # Create LocalLLM instance
    local_llm = LocalLLM(model_id="gpt2")
    
    # Mock torch.no_grad context
    with patch('promptolution.llms.local_llm.torch.no_grad') as mock_no_grad:
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock()
        
        # Call _get_response
        prompts = ["Test prompt 1", "Test prompt 2"]
        system_prompts = ["Be helpful", "Be concise"]
        responses = local_llm._get_response(prompts, system_prompts)
        
        # Verify pipeline was called
        local_llm.pipeline.assert_called_once()
        
        # Verify torch.no_grad was used
        mock_no_grad.assert_called_once()
        
        # Verify responses
        assert len(responses) == 2
        assert responses[0] == "Mock response 1"
        assert responses[1] == "Mock response 2"