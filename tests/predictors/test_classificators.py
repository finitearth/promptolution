import pytest
import numpy as np

from promptolution.predictors.classificator import FirstOccurrenceClassificator, MarkerBasedClassificator
from tests.mocks.mock_llm import MockLLM
from tests.fixtures import mock_llm_for_first_occurrence, mock_llm_for_marker_based, mock_df


def test_first_occurrence_classificator(mock_llm_for_first_occurrence, mock_df):
    """Test the FirstOccurrenceClassificator."""
    # Create classifier
    classifier = FirstOccurrenceClassificator(llm=mock_llm_for_first_occurrence, classes=mock_df["y"].values)

    # Test with multiple inputs
    xs = np.array(["I love this product!", "I hate this product!", "This product is okay."])
    prompts = ["Classify:"] * len(xs)

    # Make predictions
    predictions = classifier.predict(prompts, xs)

    # Verify shape and content
    assert predictions.shape == (3,)
    assert predictions[0] == "positive"
    assert predictions[1] == "negative"
    assert predictions[2] == "neutral"

    # Test with input that doesn't contain a class directly
    ambiguous_input = np.array(["Interesting product"])
    ambiguous_predictions = classifier.predict(prompts[0], ambiguous_input)

    # Should default to first class if no match
    assert ambiguous_predictions[0] == "positive"


def test_marker_based_classificator(mock_llm_for_marker_based, sentiment_classes):
    """Test the MarkerBasedClassificator."""
    # Create classifier
    classifier = MarkerBasedClassificator(
        llm=mock_llm_for_marker_based,
        classes=sentiment_classes,
        begin_marker="<final_answer>",
        end_marker="</final_answer>",
    )

    # Test with multiple inputs
    xs = np.array(["I love this product!", "I hate this product!", "This product is okay."])
    prompts = ["Classify:"] * len(xs)

    # Make predictions
    predictions = classifier.predict(prompts, xs)

    # Verify shape and content
    assert predictions.shape == (3,)
    assert predictions[0] == "positive"
    assert predictions[1] == "negative"
    assert predictions[2] == "neutral"

    # Test with invalid class label
    invalid_input = np.array(["Broken item"] * len(prompts))
    invalid_predictions = classifier.predict(prompts, invalid_input)

    # Should default to first class if invalid
    assert invalid_predictions[0] == "positive"


def test_marker_based_without_classes(mock_llm_for_marker_based):
    """Test MarkerBasedClassificator without predefined classes."""
    # Create classifier without classes
    classifier = MarkerBasedClassificator(
        llm=mock_llm_for_marker_based,
        classes=None,  # No class restrictions
        begin_marker="<final_answer>",
        end_marker="</final_answer>",
    )

    # Test with multiple inputs
    xs = np.array(["I love this product!", "Broken item"])
    prompts = ["Classify:"] * len(xs)

    # Make predictions
    predictions = classifier.predict(prompts, xs)

    # Verify shape and content - should accept any value between markers
    assert predictions.shape == (2,)
    assert predictions[0] == "positive"
    assert predictions[1] == "bad"  # Should accept "bad" as it's between markers


def test_multiple_prompts_with_classificators(mock_llm_for_first_occurrence, sentiment_classes):
    """Test using multiple prompts with classificators."""
    # Create classifier
    classifier = FirstOccurrenceClassificator(llm=mock_llm_for_first_occurrence, classes=sentiment_classes)

    # Test with multiple prompts
    prompts = ["Classify:", "Classify:", "Rate:", "Rate:"]
    xs = np.array(["I love this product!", "I hate this product!"] * 2)

    # Make predictions
    predictions = classifier.predict(prompts, xs)

    # Verify shape and content
    assert predictions.shape == (4,)
    assert predictions[0] == "positive"  # First prompt, first sample
    assert predictions[1] == "negative"  # First prompt, second sample
    assert predictions[2] == "positive"  # Second prompt, first sample
    assert predictions[3] == "negative"  # Second prompt, second sample


def test_sequence_return_with_classificators(mock_llm_for_marker_based, sentiment_classes):
    """Test return_seq parameter with classificators."""
    # Create classifier
    classifier = MarkerBasedClassificator(llm=mock_llm_for_marker_based, classes=sentiment_classes)

    # Test with return_seq=True
    prompts = ["Classify:"]
    xs = np.array(["I love this product!"])

    # Make predictions with sequences
    predictions, sequences = classifier.predict(prompts, xs, return_seq=True)

    # Verify predictions
    assert predictions.shape == (1,)
    assert predictions[0] == "positive"

    # Verify sequences
    assert len(sequences) == 1
    assert "I love this product!" in sequences[0]
