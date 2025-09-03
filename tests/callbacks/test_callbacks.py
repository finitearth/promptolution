import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.mocks.mock_llm import MockLLM

from promptolution.utils import (
    BestPromptCallback,
    FileOutputCallback,
    LoggerCallback,
    ProgressBarCallback,
    TokenCountCallback,
)


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer with the necessary attributes for callbacks."""
    optimizer = MagicMock()
    optimizer.prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    optimizer.scores = [0.8, 0.7, 0.6]
    optimizer.meta_llm = MockLLM()
    optimizer.meta_llm.input_token_count = 100
    optimizer.meta_llm.output_token_count = 50

    # Add predictor with LLM for TokenCountCallback
    optimizer.predictor = MagicMock()
    optimizer.predictor.llm = MockLLM()
    optimizer.predictor.llm.input_token_count = 200
    optimizer.predictor.llm.output_token_count = 100

    return optimizer


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing LoggerCallback."""
    logger = MagicMock()
    logger.critical = MagicMock()
    return logger


def test_logger_callback(mock_optimizer, mock_logger):
    """Test the LoggerCallback class."""
    callback = LoggerCallback(mock_logger)

    # Test initialization
    assert callback.logger == mock_logger
    assert callback.step == 0

    # Test on_step_end
    result = callback.on_step_end(mock_optimizer)
    assert result is True
    assert callback.step == 1

    # Verify logger was called
    assert mock_logger.critical.call_count >= 5  # Time + Step + 3 prompts

    # Test on_train_end
    result = callback.on_train_end(mock_optimizer)
    assert result is True

    # Test on_train_end with logs
    result = callback.on_train_end(mock_optimizer, logs="Test logs")
    assert result is True

    # Verify logger was called with logs
    mock_logger.critical.assert_any_call(mock_logger.critical.call_args_list[-1][0][0])


def test_file_output_callback_csv(mock_optimizer, tmpdir):
    """Test the FileOutputCallback with CSV output."""
    output_dir = str(tmpdir.mkdir("test_output"))
    callback = FileOutputCallback(dir=output_dir, file_type="csv")

    # Test initialization
    assert callback.file_type == "csv"
    assert callback.path == os.path.join(output_dir, "step_results.csv")
    assert callback.step == 0

    # Test on_step_end - first step
    result = callback.on_step_end(mock_optimizer)
    assert result is True
    assert callback.step == 1

    # Verify file was created
    assert os.path.exists(callback.path)

    # Read the CSV file and verify content
    df = pd.read_csv(callback.path)
    assert len(df) == 3  # 3 prompts
    assert "step" in df.columns
    assert "score" in df.columns
    assert "prompt" in df.columns
    assert all(df["step"] == 1)

    # Test on_step_end - second step
    result = callback.on_step_end(mock_optimizer)
    assert result is True
    assert callback.step == 2

    # Verify file was updated
    df = pd.read_csv(callback.path)
    assert len(df) == 6  # 3 prompts × 2 steps
    assert set(df["step"]) == {1, 2}


def test_file_output_callback_parquet(mock_optimizer, tmpdir):
    """Test the FileOutputCallback with Parquet output."""
    output_dir = str(tmpdir.mkdir("test_output_parquet"))
    callback = FileOutputCallback(dir=output_dir, file_type="parquet")

    # Test initialization
    assert callback.file_type == "parquet"
    assert callback.path == os.path.join(output_dir, "step_results.parquet")

    # Test on_step_end - first step
    result = callback.on_step_end(mock_optimizer)
    assert result is True

    # Verify file was created
    assert os.path.exists(callback.path)

    # Read the Parquet file and verify content
    df = pd.read_parquet(callback.path)
    assert len(df) == 3  # 3 prompts
    assert "step" in df.columns
    assert "score" in df.columns
    assert "prompt" in df.columns
    assert all(df["step"] == 1)


def test_file_output_callback_invalid_type():
    """Test FileOutputCallback with invalid file type."""
    with pytest.raises(ValueError):
        FileOutputCallback(dir="test", file_type="invalid")


def test_best_prompt_callback(mock_optimizer):
    """Test the BestPromptCallback class."""
    callback = BestPromptCallback()

    # Test initialization
    assert callback.best_prompt == ""
    assert callback.best_score == -99999

    # Test on_step_end
    result = callback.on_step_end(mock_optimizer)
    assert result is True
    assert callback.best_prompt == "Prompt 1"
    assert callback.best_score == 0.8

    # Test with better score
    mock_optimizer.scores = [0.9, 0.7, 0.6]
    mock_optimizer.prompts = ["Better Prompt", "Prompt 2", "Prompt 3"]

    result = callback.on_step_end(mock_optimizer)
    assert result is True
    assert callback.best_prompt == "Better Prompt"
    assert callback.best_score == 0.9

    # Test with worse score
    mock_optimizer.scores = [0.7, 0.6, 0.5]
    mock_optimizer.prompts = ["Worse Prompt", "Prompt 2", "Prompt 3"]

    result = callback.on_step_end(mock_optimizer)
    assert result is True
    assert callback.best_prompt == "Better Prompt"  # Unchanged
    assert callback.best_score == 0.9  # Unchanged

    # Test get_best_prompt
    best_prompt, best_score = callback.get_best_prompt()
    assert best_prompt == "Better Prompt"
    assert best_score == 0.9


def test_progress_bar_callback():
    """Test the ProgressBarCallback class."""
    with patch("promptolution.utils.callbacks.tqdm") as mock_tqdm:
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        # Create callback
        callback = ProgressBarCallback(total_steps=10)

        # Verify tqdm was called with correct arguments
        mock_tqdm.assert_called_once_with(total=10)

        # Test on_step_end
        result = callback.on_step_end(None)
        assert result is True
        mock_pbar.update.assert_called_once_with(1)

        # Test on_train_end
        result = callback.on_train_end(None)
        assert result is True
        mock_pbar.close.assert_called_once()


def test_token_count_callback(mock_optimizer):
    """Test the TokenCountCallback class."""
    # Test with input tokens
    callback = TokenCountCallback(max_tokens_for_termination=300, token_type_for_termination="input_tokens")

    # Replace get_token_count with our own function
    def get_token_count_under_limit():
        return {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}

    # Replace the method directly
    mock_optimizer.predictor.llm.get_token_count = get_token_count_under_limit

    # Should continue as we're below the max
    result = callback.on_step_end(mock_optimizer)
    assert result is True

    # Now replace with a function that exceeds the limit
    def get_token_count_over_limit():
        return {"input_tokens": 400, "output_tokens": 100, "total_tokens": 500}  # Over the limit

    mock_optimizer.predictor.llm.get_token_count = get_token_count_over_limit

    # Should stop
    result = callback.on_step_end(mock_optimizer)
    assert result is False

    # Test with output tokens
    callback = TokenCountCallback(max_tokens_for_termination=150, token_type_for_termination="output_tokens")

    # Use the same approach - replace with function
    def get_token_count_output_under_limit():
        return {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}  # Under the limit

    mock_optimizer.predictor.llm.get_token_count = get_token_count_output_under_limit

    # Should continue as we're below the max
    result = callback.on_step_end(mock_optimizer)
    assert result is True

    # Now replace with a function that exceeds the output tokens limit
    def get_token_count_output_over_limit():
        return {"input_tokens": 200, "output_tokens": 200, "total_tokens": 400}  # Over the limit

    mock_optimizer.predictor.llm.get_token_count = get_token_count_output_over_limit

    # Should stop
    result = callback.on_step_end(mock_optimizer)
    assert result is False
