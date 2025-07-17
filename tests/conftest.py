"""Fixtures for testing."""

import pandas as pd
import pytest
from mocks.mock_llm import MockLLM
from mocks.mock_predictor import MockPredictor
from mocks.mock_task import MockTask

from promptolution.tasks import ClassificationTask
from promptolution.utils import ExperimentConfig


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

    def score_function(pred):
        # Prefer longer prompts for testing purposes
        return len(pred)

    return MockTask(predetermined_scores=score_function)


@pytest.fixture
def mock_meta_llm():
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
def mock_downstream_llm():
    """Fixture providing a MockLLM configured for downstream task responses."""
    # Create a list of responses that already include prompt tags
    responses = [
        "This review is not negative, so my answer is <final_answer>positive</final_answer>",
        "This review is not positive, so my answer is <final_answer>negative</final_answer>",
        "This review is neither positive nor negative, so my answer is <final_answer>neutral</final_answer>",
        "Pfff hard to say, <final_answer>I dont know</final_answer>",
    ]

    # Simply initialize the MockLLM with the predetermined responses
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def mock_predictor(mock_downstream_llm):
    """Fixture providing a MockPredictor for testing."""
    mock_predictor = MockPredictor(llm=mock_downstream_llm)
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
def mock_classification_task_with_subsampling(mock_df):
    """Fixture providing a ClassificationTask instance with subsampling."""
    return ClassificationTask(
        df=mock_df,
        task_description="Sentiment classification task",
        x_column="x",
        y_column="y",
        eval_strategy="subsample",
        n_subsamples=2,
    )
