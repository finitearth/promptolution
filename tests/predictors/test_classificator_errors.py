import pytest
import numpy as np

from promptolution.predictors.classificator import FirstOccurrenceClassificator, MarkerBasedClassificator
from tests.mocks.mock_llm import MockLLM


def test_invalid_class_labels():
    """Test that classifier raises an assertion error for invalid class labels."""
    mock_llm = MockLLM()
    
    # Classes should be lowercase
    invalid_classes = ["Positive", "Neutral", "Negative"]
    
    # Should raise an assertion error
    with pytest.raises(AssertionError):
        FirstOccurrenceClassificator(llm=mock_llm, classes=invalid_classes)
    
    with pytest.raises(AssertionError):
        MarkerBasedClassificator(llm=mock_llm, classes=invalid_classes)


def test_marker_based_missing_markers():
    """Test MarkerBasedClassificator behavior when markers are missing."""
    mock_llm = MockLLM(predetermined_responses={
        "Classify: Missing markers": "This response doesn't have the markers at all."
    })
    
    classifier = MarkerBasedClassificator(
        llm=mock_llm,
        classes=["positive", "neutral", "negative"]
    )
    
    # When markers are missing, it should default to first class
    prompts = ["Classify:"]
    xs = np.array(["Missing markers"])
    predictions = classifier.predict(prompts, xs)
    
    assert predictions[0] == "positive"  # Should default to first class