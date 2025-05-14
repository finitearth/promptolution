import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_vllm_dependencies():
    """Set up comprehensive mocks for VLLM dependencies."""
    # Mock the key components
    with patch('vllm.LLM') as mock_llm_class, \
         patch('vllm.SamplingParams') as mock_sampling_params, \
         patch('transformers.AutoTokenizer') as mock_tokenizer_class:

        # Create and configure mock LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        
        # Configure LLM engine with cache config for batch size calculation
        mock_cache_config = MagicMock()
        mock_cache_config.num_gpu_blocks = 100
        mock_cache_config.block_size = 16
        
        mock_executor = MagicMock()
        mock_executor.cache_config = mock_cache_config
        
        mock_engine = MagicMock()
        mock_engine.model_executor = mock_executor
        
        mock_llm.llm_engine = mock_engine
        
        # Set up the generate method to return appropriate number of responses
        def mock_generate_side_effect(prompts_list, *args, **kwargs):
            """Return one output per input prompt"""
            return [
                MagicMock(outputs=[MagicMock(text=f"Mocked response for prompt {i}")])
                for i, _ in enumerate(prompts_list)
            ]
        
        # Use side_effect instead of return_value for dynamic behavior
        mock_llm.generate.side_effect = mock_generate_side_effect
        
        # Configure mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.apply_chat_template.return_value = "<mocked_chat_template>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        yield {
            'llm_class': mock_llm_class,
            'llm': mock_llm,
            'tokenizer_class': mock_tokenizer_class,
            'tokenizer': mock_tokenizer,
            'sampling_params': mock_sampling_params
        }


def test_vllm_get_response(mock_vllm_dependencies):
    """Test that VLLM._get_response works correctly with explicit batch_size."""
    # Create VLLM instance with explicit batch_size to avoid calculation
    vllm = VLLM(
        model_id="mock-model",
        batch_size=4  # Set an explicit batch_size to avoid computation
    )
    
    # Call get_response
    prompts = ["Test prompt 1", "Test prompt 2"]
    system_prompts = ["Be helpful", "Be concise"]
    responses = vllm._get_response(prompts, system_prompts)
    
    # Verify tokenizer was used correctly
    assert mock_vllm_dependencies['tokenizer'].apply_chat_template.call_count == 2
    
    # Verify LLM generate was called
    mock_vllm_dependencies['llm'].generate.assert_called_once()
    
    # Verify responses
    assert len(responses) == 2
    assert responses[0] == "Mocked response for prompt 0"
    assert responses[1] == "Mocked response for prompt 1"


def test_vllm_with_auto_batch_size(mock_vllm_dependencies):
    """Test VLLM with automatic batch size calculation."""
    # Create VLLM instance with batch_size=None to trigger auto calculation
    vllm = VLLM(
        model_id="mock-model",
        batch_size=None,
        max_model_len=2048
    )
    
    # Force a non-zero batch size
    mock_vllm_dependencies['llm'].llm_engine.model_executor.cache_config.num_gpu_blocks = 1000
    
    # Create a new instance to recalculate batch size
    vllm = VLLM(
        model_id="mock-model",
        batch_size=None,
        max_model_len=2048
    )
    
    # Verify batch_size is greater than zero
    assert vllm.batch_size > 0, "Batch size should be greater than zero"
    
    # Test with a single prompt
    prompts = ["Test prompt"]
    system_prompts = ["Be helpful"]
    responses = vllm._get_response(prompts, system_prompts)
    
    # Verify we get exactly one response for one prompt
    assert len(responses) == 1
    assert responses[0] == "Mocked response for prompt 0"
