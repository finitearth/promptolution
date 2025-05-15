import pytest
from mocks.mock_llm import MockLLM
from mocks.mock_task import MockTask

from promptolution.config import ExperimentConfig
from mocks.mock_predictor import MockPredictor

import pandas as pd


@pytest.fixture
def initial_prompts():
    """Fixture providing a set of initial prompts for testing."""
    return [
        "Classify the following text as positive or negative.",
        "Determine if the sentiment of the text is positive or negative.",
        "Is the following text positive or negative?",
        "Analyze the sentiment of this text and categorize as positive or negative.",
    ]


@pytest.fixture
def experiment_config():
    """Fixture providing a basic experiment configuration."""
    return ExperimentConfig(optimizer_name="test_optimizer", n_steps=3, population_size=3, random_seed=42)


@pytest.fixture
def mock_task():
    """Fixture providing a MockTask with predetermined scoring behavior."""

    # A function that generates scores based on the prompt
    def score_function(prompt):
        # Prefer longer prompts for testing purposes
        return min(0.9, 0.5 + 0.01 * len(prompt))

    return MockTask(predetermined_scores=score_function)


@pytest.fixture
def meta_llm_mock():
    """Fixture providing a MockLLM configured for meta-prompt responses."""
    # Create a list of responses that already include prompt tags
    responses = [
        "<prompt>Meta-generated prompt for input 0</prompt>",
        "<prompt>Meta-generated prompt for input 1</prompt>",
        "<prompt>Combined prompt</prompt>",
        "<prompt>Improved version of prompt</prompt>",
        "<prompt>Default meta-response</prompt>",
    ]

    # Simply initialize the MockLLM with the predetermined responses
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def downstream_llm_mock():
    """Fixture providing a MockLLM configured for downstream task responses."""
    # Create a list of responses that already include prompt tags
    responses = ["positive", "negative", "neutral"]

    # Simply initialize the MockLLM with the predetermined responses
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def mock_predictor(downstream_llm_mock):
    """Fixture providing a MockPredictor for testing."""
    mock_predictor = MockPredictor(llm=downstream_llm_mock)
    return mock_predictor


@pytest.fixture
def mock_df():
    """Fixture providing a DataFrame with few-shot examples for CAPO."""
    return pd.DataFrame(
        {
            "x": ["This is excellent!", "I hate this product.", "Average experience."],
            "y": ["positive", "negative", "neutral"],
        }
    )


@pytest.fixture
def mock_llm_for_first_occurrence():
    """Fixture providing a MockLLM for FirstOccurrenceClassificator testing."""
    responses = [
        "This text expresses positive sentiment about the product.",
        "The sentiment here is clearly negative regarding the item.",
        "The text shows a neutral stance on the product.",
        "The sentiment is hard to determine, but not strongly leaning.",
    ]
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def mock_llm_for_marker_based():
    """Fixture providing a MockLLM for MarkerBasedClassificator testing."""
    responses = [
        "Let me analyze this... <final_answer>positive</final_answer>",
        "I can detect that this is <final_answer>negative</final_answer>",
        "After consideration <final_answer>neutral</final_answer> seems appropriate",
        "Not entirely clear but <final_answer>neutral</final_answer> is best",
        "This shows <final_answer>bad</final_answer> sentiment",  # Invalid class for testing default
    ]
    return MockLLM(predetermined_responses=responses)
