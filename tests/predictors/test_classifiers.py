import numpy as np
import pytest

from promptolution.helpers import FirstOccurrenceClassifier, MarkerBasedClassifier


def test_first_occurrence_classifier(mock_downstream_llm, mock_df):
    """Test the FirstOccurrenceClassifier."""
    # Create classifier
    classifier = FirstOccurrenceClassifier(llm=mock_downstream_llm, classes=mock_df["y"].values)

    # Test with multiple inputs
    xs = np.array(["I love this product!", "I hate this product!", "This product is okay.", "ja ne"])
    prompts = ["Classify:"] * len(xs)

    # Make predictions
    predictions = classifier.predict(prompts, xs)

    # Verify shape and content
    assert predictions.shape == (4,)
    assert predictions[0] == "negative"
    assert predictions[1] == "positive"
    assert predictions[2] == "positive"
    assert predictions[3] == "positive"


def test_marker_based_classifier(mock_downstream_llm, mock_df):
    """Test the MarkerBasedClassifier."""
    # Create classifier
    classifier = MarkerBasedClassifier(
        llm=mock_downstream_llm,
        classes=mock_df["y"].values,
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


def test_marker_based_without_classes(mock_downstream_llm):
    """Test MarkerBasedClassifier without predefined classes."""
    # Create classifier without classes
    classifier = MarkerBasedClassifier(
        llm=mock_downstream_llm,
        classes=None,  # No class restrictions
        begin_marker="<final_answer>",
        end_marker="</final_answer>",
    )

    # Test with multiple inputs
    xs = np.array(["I love this product!", "Broken item", "Cant complain", "I dont know"])
    prompts = ["Classify:"] * len(xs)

    # Make predictions
    predictions = classifier.predict(prompts, xs)

    # Verify shape and content - should accept any value between markers
    assert predictions.shape == (4,)
    assert predictions[0] == "positive"
    assert predictions[1] == "negative"
    assert predictions[2] == "neutral"
    assert predictions[3] == "i dont know"


def test_multiple_prompts_with_classifiers(mock_downstream_llm, mock_df):
    """Test using multiple prompts with classifiers."""
    # Create classifier
    classifier = FirstOccurrenceClassifier(llm=mock_downstream_llm, classes=mock_df["y"].values)

    # Test with multiple prompts
    prompts = ["Classify:", "Classify:", "Rate:", "Rate:"]
    xs = np.array(["I love this product!", "I hate this product!"] * 2)

    # Make predictions
    predictions = classifier.predict(prompts, xs)

    # Verify shape and content
    assert predictions.shape == (4,)
    assert predictions[0] == "negative"
    assert predictions[1] == "positive"
    assert predictions[2] == "positive"
    assert predictions[3] == "positive"


def test_sequence_return_with_classifiers(mock_downstream_llm, mock_df):
    """Test return_seq parameter with classifiers."""
    # Create classifier
    classifier = MarkerBasedClassifier(llm=mock_downstream_llm, classes=mock_df["y"].values)

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


def test_invalid_class_labels(mock_downstream_llm):
    """Test that classifier raises an assertion error for invalid class labels."""
    # Classes should be lowercase
    invalid_classes = ["Positive", "Neutral", "Negative"]

    # Should raise an assertion error
    with pytest.raises(AssertionError):
        FirstOccurrenceClassifier(llm=mock_downstream_llm, classes=invalid_classes)

    with pytest.raises(AssertionError):
        MarkerBasedClassifier(llm=mock_downstream_llm, classes=invalid_classes)


def test_marker_based_missing_markers(mock_downstream_llm):
    """Test MarkerBasedClassifier behavior when markers are missing."""
    classifier = MarkerBasedClassifier(llm=mock_downstream_llm, classes=["will", "not", "be", "used"])

    # When markers are missing, it should default to first class
    prompts = ["Classify:"]
    xs = np.array(["Missing markers"])
    predictions = classifier.predict(prompts, xs)

    assert predictions[0] == "will"  # Should default to first class
