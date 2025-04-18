# tests/conftest.py
import pytest
import numpy as np

from promptolution.config import ExperimentConfig
from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_task import MockTask
from tests.mocks.mock_predictor import MockPredictor

from tests.mocks.mock_optimizer import MockOptimizer

@pytest.fixture
def base_optimizer_config():
    """Fixture providing a basic optimizer configuration."""
    return ExperimentConfig(
        optimizer_name="test_optimizer",
        n_steps=5,
        population_size=8,
        random_seed=42
    )


@pytest.fixture
def initial_prompts():
    """Fixture providing initial prompts for optimizer testing."""
    return [
        "Classify the sentiment of the text.",
        "Determine if the text is positive or negative.",
        "Analyze the sentiment in the following text."
    ]


@pytest.fixture
def dummy_task():
    """Fixture providing a dummy task for optimizer testing."""
    task = MockTask(predetermined_scores=[0.6, 0.7, 0.8])
    return task


@pytest.fixture
def dummy_predictor():
    """Fixture providing a dummy predictor for optimizer testing."""
    return MockPredictor(
        classes=["positive", "neutral", "negative"]
    )


@pytest.fixture
def dummy_llm():
    """Fixture providing a dummy LLM for optimizer testing."""
    llm = MockLLM()
    llm._get_response = lambda prompts, system_prompts: [
        "<prompt>Generated prompt for test</prompt>" for _ in prompts
    ]
    return llm


@pytest.fixture
def mock_optimizer():
    """Fixture providing a MockOptimizer for testing."""
    return MockOptimizer()