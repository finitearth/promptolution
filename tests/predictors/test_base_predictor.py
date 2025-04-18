import pytest
import numpy as np

from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_predictor import MockPredictor


@pytest.fixture
def mock_llm_for_predictor():
    """Fixture providing a MockLLM with predetermined responses for predictor testing."""
    responses = {
        "Classify this text:\nI love this product!": "The sentiment is positive.",
        "Classify this text:\nI hate this product!": "The sentiment is negative.",
        "Classify this text:\nThis product is okay.": "The sentiment is neutral."
    }
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def mock_predictor_with_llm(mock_llm_for_predictor):
    """Fixture providing a MockPredictor with a MockLLM."""
    predictions = {
        "The sentiment is positive.": "positive",
        "The sentiment is negative.": "negative",
        "The sentiment is neutral.": "neutral"  # Make sure this mapping exists
    }
    classes = ["positive", "neutral", "negative"]
    
    return MockPredictor(
        llm=mock_llm_for_predictor,
        classes=classes,
        predetermined_predictions=predictions
    )


def test_predictor_predict_flow(mock_predictor_with_llm):
    """Test the basic prediction flow from prompt to final prediction."""
    # Input data
    prompts = ["Classify this text:"]
    xs = np.array(["I love this product!", "I hate this product!"])
    
    # Call predict
    predictions = mock_predictor_with_llm.predict(prompts, xs)
    
    # Verify shape and content of predictions
    assert predictions.shape == (1, 2)
    assert predictions[0, 0] == "positive"
    assert predictions[0, 1] == "negative"
    
    # Verify LLM was called with correct prompts
    assert len(mock_predictor_with_llm.llm.call_history) == 1
    assert mock_predictor_with_llm.llm.call_history[0]['prompts'] == [
        "Classify this text:\nI love this product!", 
        "Classify this text:\nI hate this product!"
    ]


def test_predictor_with_return_seq(mock_predictor_with_llm):
    """Test prediction with return_seq=True."""
    # Input data
    prompts = ["Classify this text:"]
    xs = np.array(["This product is okay."])
    
    # Call predict with return_seq=True
    predictions, sequences = mock_predictor_with_llm.predict(prompts, xs, return_seq=True)
    
    # Verify predictions
    assert predictions.shape == (1, 1)
    assert predictions[0, 0] == "neutral"
    
    # Verify sequences
    assert len(sequences) == 1
    assert isinstance(sequences[0], np.ndarray)
    assert "This product is okay." in sequences[0][0]


def test_multiple_prompts(mock_predictor_with_llm):
    """Test prediction with multiple prompts."""
    # Input data with multiple prompts
    prompts = ["Classify this text:", "Rate this text:"]
    xs = np.array(["I love this product!"])
    
    # Mock LLM responses for the second prompt
    mock_predictor_with_llm.llm.predetermined_responses.update({
        "Rate this text:\nI love this product!": "The rating is 5/5."
    })
    
    # Add mapping for the new response
    mock_predictor_with_llm.predetermined_predictions.update({
        "The rating is 5/5.": "positive"
    })
    
    # Call predict
    predictions = mock_predictor_with_llm.predict(prompts, xs)
    
    # Verify shape and content
    assert predictions.shape == (2, 1)
    assert predictions[0, 0] == "positive"  # First prompt result
    assert predictions[1, 0] == "positive"  # Second prompt result