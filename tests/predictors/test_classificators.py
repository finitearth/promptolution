import pytest
import numpy as np

from promptolution.predictors.classificator import FirstOccurrenceClassificator, MarkerBasedClassificator
from tests.mocks.mock_llm import MockLLM


@pytest.fixture
def sentiment_classes():
    """Fixture providing sentiment classes for testing."""
    return ["positive", "neutral", "negative"]


@pytest.fixture
def mock_llm_for_first_occurrence():
    """Fixture providing a MockLLM for FirstOccurrenceClassificator testing."""
    responses = {
        "Classify:\nI love this product!": "This text expresses positive sentiment about the product.",
        "Classify:\nI hate this product!": "The sentiment here is clearly negative regarding the item.",
        "Classify:\nThis product is okay.": "The text shows a neutral stance on the product.",
        "Classify:\nInteresting product": "The sentiment is hard to determine, but not strongly leaning."
    }
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def mock_llm_for_marker_based():
    """Fixture providing a MockLLM for MarkerBasedClassificator testing."""
    responses = {
        "Classify:\nI love this product!": "Let me analyze this... <final_answer>positive</final_answer>",
        "Classify:\nI hate this product!": "I can detect that this is <final_answer>negative</final_answer> sentiment",
        "Classify:\nThis product is okay.": "After consideration <final_answer>neutral</final_answer> seems appropriate",
        "Classify:\nInteresting product": "Not entirely clear but <final_answer>neutral</final_answer> is best",
        "Classify:\nBroken item": "This shows <final_answer>bad</final_answer> sentiment" # Invalid class for testing default
    }
    return MockLLM(predetermined_responses=responses)


def test_first_occurrence_classificator(mock_llm_for_first_occurrence, sentiment_classes):
    """Test the FirstOccurrenceClassificator."""
    # Create classifier
    classifier = FirstOccurrenceClassificator(
        llm=mock_llm_for_first_occurrence,
        classes=sentiment_classes
    )
    
    # Test with multiple inputs
    prompts = ["Classify:"]
    xs = np.array(["I love this product!", "I hate this product!", "This product is okay."])
    
    # Make predictions
    predictions = classifier.predict(prompts, xs)
    
    # Verify shape and content
    assert predictions.shape == (1, 3)
    assert predictions[0, 0] == "positive"
    assert predictions[0, 1] == "negative"
    assert predictions[0, 2] == "neutral"
    
    # Test with input that doesn't contain a class directly
    ambiguous_input = np.array(["Interesting product"])
    ambiguous_predictions = classifier.predict(prompts, ambiguous_input)
    
    # Should default to first class if no match
    assert ambiguous_predictions[0, 0] == "positive"


def test_marker_based_classificator(mock_llm_for_marker_based, sentiment_classes):
    """Test the MarkerBasedClassificator."""
    # Create classifier
    classifier = MarkerBasedClassificator(
        llm=mock_llm_for_marker_based,
        classes=sentiment_classes,
        begin_marker="<final_answer>",
        end_marker="</final_answer>"
    )
    
    # Test with multiple inputs
    prompts = ["Classify:"]
    xs = np.array(["I love this product!", "I hate this product!", "This product is okay."])
    
    # Make predictions
    predictions = classifier.predict(prompts, xs)
    
    # Verify shape and content
    assert predictions.shape == (1, 3)
    assert predictions[0, 0] == "positive"
    assert predictions[0, 1] == "negative"
    assert predictions[0, 2] == "neutral"
    
    # Test with invalid class label
    invalid_input = np.array(["Broken item"])
    invalid_predictions = classifier.predict(prompts, invalid_input)
    
    # Should default to first class if invalid
    assert invalid_predictions[0, 0] == "positive"


def test_marker_based_without_classes(mock_llm_for_marker_based):
    """Test MarkerBasedClassificator without predefined classes."""
    # Create classifier without classes
    classifier = MarkerBasedClassificator(
        llm=mock_llm_for_marker_based,
        classes=None,  # No class restrictions
        begin_marker="<final_answer>",
        end_marker="</final_answer>"
    )
    
    # Test with multiple inputs
    prompts = ["Classify:"]
    xs = np.array(["I love this product!", "Broken item"])
    
    # Make predictions
    predictions = classifier.predict(prompts, xs)
    
    # Verify shape and content - should accept any value between markers
    assert predictions.shape == (1, 2)
    assert predictions[0, 0] == "positive"
    assert predictions[0, 1] == "bad"  # Should accept "bad" as it's between markers


def test_multiple_prompts_with_classificators(mock_llm_for_first_occurrence, sentiment_classes):
    """Test using multiple prompts with classificators."""
    # Create classifier
    classifier = FirstOccurrenceClassificator(
        llm=mock_llm_for_first_occurrence,
        classes=sentiment_classes
    )
    
    # Add responses for a second prompt
    mock_llm_for_first_occurrence.predetermined_responses.update({
        "Rate:\nI love this product!": "This deserves a positive rating.",
        "Rate:\nI hate this product!": "I would rate this as negative."
    })
    
    # Test with multiple prompts
    prompts = ["Classify:", "Rate:"]
    xs = np.array(["I love this product!", "I hate this product!"])
    
    # Make predictions
    predictions = classifier.predict(prompts, xs)
    
    # Verify shape and content
    assert predictions.shape == (2, 2)  # (n_prompts, n_samples)
    assert predictions[0, 0] == "positive"  # First prompt, first sample
    assert predictions[0, 1] == "negative"  # First prompt, second sample
    assert predictions[1, 0] == "positive"  # Second prompt, first sample
    assert predictions[1, 1] == "negative"  # Second prompt, second sample


def test_sequence_return_with_classificators(mock_llm_for_marker_based, sentiment_classes):
    """Test return_seq parameter with classificators."""
    # Create classifier
    classifier = MarkerBasedClassificator(
        llm=mock_llm_for_marker_based,
        classes=sentiment_classes
    )
    
    # Test with return_seq=True
    prompts = ["Classify:"]
    xs = np.array(["I love this product!"])
    
    # Make predictions with sequences
    predictions, sequences = classifier.predict(prompts, xs, return_seq=True)
    
    # Verify predictions
    assert predictions.shape == (1, 1)
    assert predictions[0, 0] == "positive"
    
    # Verify sequences
    assert len(sequences) == 1
    assert len(sequences[0]) == 1
    assert "I love this product!" in sequences[0][0]
    assert "Let me analyze this..." in sequences[0][0]