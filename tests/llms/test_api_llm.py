# tests/llms/test_api_llm.py
import pytest
from unittest.mock import MagicMock, patch

from promptolution.llms.api_llm import APILLM


def test_api_llm_initialization():
    """Test that APILLM initializes correctly."""
    # Create patches for all dependencies
    with patch('promptolution.llms.api_llm.AsyncOpenAI') as mock_client_class, \
         patch('promptolution.llms.api_llm.asyncio') as mock_asyncio:
        
        # Configure the mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_semaphore = MagicMock()
        mock_asyncio.Semaphore.return_value = mock_semaphore
        
        # Create APILLM instance
        api_llm = APILLM(
            api_url="https://api.example.com",
            llm="gpt-4",
            token="test-token",
            max_concurrent_calls=10
        )
        
        # Verify AsyncOpenAI was called correctly
        mock_client_class.assert_called_once()
        args, kwargs = mock_client_class.call_args
        assert kwargs["base_url"] == "https://api.example.com"
        assert kwargs["api_key"] == "test-token"
        
        # Verify semaphore was created
        mock_asyncio.Semaphore.assert_called_once_with(10)
        
        # Verify instance attributes
        assert api_llm.api_url == "https://api.example.com"
        assert api_llm.llm == "gpt-4"
        assert api_llm.max_concurrent_calls == 10