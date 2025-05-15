import pytest
import numpy as np

from tests.mocks.mock_predictor import MockPredictor
from tests.fixtures import downstream_llm_mock, mock_task, mock_predictor


def test_predictor_predict_flow(downstream_llm_mock):
    """Test the basic prediction flow from prompt to final prediction."""
    # Input data
    xs = np.array(["I love this product!", "I hate this product!", "Is a ok product!"])
    prompts = ["Classify this text:"] * len(xs)

    # Call predict
    predictions = downstream_llm_mock.predict(prompts, xs)
    # Verify shape and content of predictions
    assert predictions.shape == (3,)
    assert predictions[0] == "positive"
    assert predictions[1] == "negative"
    assert predictions[2] == "neutral"

    # Verify LLM was called with correct prompts
    assert len(downstream_llm_mock.llm.call_history) == 1
    assert downstream_llm_mock.llm.call_history[0]["prompts"] == [
        "Classify this text:\nI love this product!",
        "Classify this text:\nI hate this product!",
        "Classify this text:\nIs a ok product!",
    ]


def test_predictor_with_return_seq(mock_predictor):
    """Test prediction with return_seq=True."""
    # Input data
    prompts = ["Classify this text:"]
    xs = np.array(["This product is okay."])

    # Call predict with return_seq=True
    predictions, sequences = mock_predictor.predict(prompts, xs, return_seq=True)

    # Verify predictions
    assert predictions.shape == (1,)
    assert predictions[0] == "neutral"

    # Verify sequences
    assert len(sequences) == 1
    assert isinstance(sequences, np.ndarray)
    assert "This product is okay." in sequences[0]
