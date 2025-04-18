import pytest
import numpy as np

from promptolution.predictors.classificator import FirstOccurrenceClassificator
from tests.mocks.mock_llm import MockLLM


@pytest.fixture
def mock_llm_with_system_prompts():
    """Fixture providing a MockLLM that respects system prompts."""
    responses = {
        # Responses with default system prompt
        "Classify: I love this product!": "This is positive feedback.",
        
        # Responses with custom system prompts
        ("Classify: I love this product!", "Be specific"): "This text shows a clear positive sentiment.",
        ("Classify: I love this product!", "Be brief"): "positive",
        
        # Additional examples
        ("Classify: I hate this product!", "Be specific"): "The text expresses strong negative sentiment.",
        ("Classify: I hate this product!", "Be brief"): "negative"
    }
    return MockLLM(predetermined_responses=responses)


def test_classificator_with_system_prompts(mock_llm_with_system_prompts):
    """Test classificators with custom system prompts."""
    classes = ["positive", "neutral", "negative"]
    
    # Create classifier
    classifier = FirstOccurrenceClassificator(
        llm=mock_llm_with_system_prompts,
        classes=classes
    )
    
    # Test with default system prompt
    prompts = ["Classify:"]
    xs = np.array(["I love this product!"])
    default_predictions = classifier.predict(prompts, xs)
    assert default_predictions[0, 0] == "positive"
    
    # Test with custom system prompt
    custom_system_prompts = ["Be specific"]
    specific_predictions = classifier.predict(prompts, xs, system_prompts=custom_system_prompts)
    
    # The prediction should be the same, but the LLM would have received a different system prompt
    assert specific_predictions[0, 0] == "positive"
    
    # Verify system prompt was passed correctly
    assert mock_llm_with_system_prompts.call_history[-1]['system_prompts'] == custom_system_prompts
    
    # Test with "be brief" system prompt
    brief_system_prompts = ["Be brief"]
    brief_predictions = classifier.predict(prompts, xs, system_prompts=brief_system_prompts)
    assert brief_predictions[0, 0] == "positive"