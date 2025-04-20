import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.predictors.classificator import FirstOccurrenceClassificator
from tests.mocks.mock_llm import MockLLM


@pytest.fixture
def sample_classification_df():
    """Fixture providing a sample classification DataFrame for testing."""
    data = {
        'x': [
            "This product is amazing!",
            "I'm disappointed with this purchase.",
            "The quality is average, nothing special.",
            "Worst product ever, avoid at all costs!",
            "Decent product, does what it's supposed to."
        ],
        'y': [
            "positive",
            "negative",
            "neutral",
            "negative", 
            "positive"
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_llm_for_classification():
    """Fixture providing a MockLLM with predetermined responses for classification."""
    responses = {
        "Classify sentiment: This product is amazing!": "This shows positive sentiment.",
        "Classify sentiment: I'm disappointed with this purchase.": "This is negative feedback.",
        "Classify sentiment: The quality is average, nothing special.": "This is a neutral review.",
        "Classify sentiment: Worst product ever, avoid at all costs!": "Very negative review.",
        "Classify sentiment: Decent product, does what it's supposed to.": "Somewhat positive feedback.",
        
        "Rate the text: This product is amazing!": "Rating: positive",
        "Rate the text: I'm disappointed with this purchase.": "Rating: negative",
        "Rate the text: The quality is average, nothing special.": "Rating: neutral",
        "Rate the text: Worst product ever, avoid at all costs!": "Rating: negative",
        "Rate the text: Decent product, does what it's supposed to.": "Rating: positive"
    }
    return MockLLM(predetermined_responses=responses)


@pytest.fixture
def classification_task(sample_classification_df):
    """Fixture providing a ClassificationTask instance."""
    initial_prompts = ["Classify sentiment:", "Rate the text:"]
    return ClassificationTask(
        df=sample_classification_df,
        description="Sentiment classification task",
        initial_prompts=initial_prompts,
        x_column="x",
        y_column="y",
        metric=accuracy_score
    )


@pytest.fixture
def classifier_predictor(mock_llm_for_classification):
    """Fixture providing a FirstOccurrenceClassificator instance."""
    return FirstOccurrenceClassificator(
        llm=mock_llm_for_classification,
        classes=["positive", "neutral", "negative"]
    )


def test_classification_task_initialization(sample_classification_df):
    """Test that ClassificationTask initializes correctly."""
    task = ClassificationTask(
        df=sample_classification_df,
        description="Sentiment classification task",
        initial_prompts=["Classify:"],
        x_column="x",
        y_column="y"
    )
    
    # Verify attributes
    assert task.description == "Sentiment classification task"
    assert len(task.classes) == 3
    assert set(task.classes) == set(["positive", "neutral", "negative"])
    assert len(task.xs) == 5
    assert len(task.ys) == 5
    assert task.initial_prompts == ["Classify:"]
    assert task.metric == accuracy_score


def test_task_evaluate(classification_task, classifier_predictor):
    """Test the evaluate method of ClassificationTask."""
    # Evaluate with a single prompt
    prompts = ["Classify sentiment:"]
    scores = classification_task.evaluate(prompts, classifier_predictor)
    
    # Verify scores
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)  # One score per prompt
    assert 0 <= scores[0] <= 1  # Score should be between 0 and 1
    
    # Evaluate with multiple prompts
    prompts = ["Classify sentiment:", "Rate the text:"]
    scores = classification_task.evaluate(prompts, classifier_predictor)
    
    # Verify scores for multiple prompts
    assert scores.shape == (2,)  # Two scores, one per prompt
    assert all(0 <= score <= 1 for score in scores)


def test_task_evaluate_with_subsampling(classification_task, classifier_predictor):
    """Test the evaluate method with subsampling."""
    prompts = ["Classify sentiment:"]
    
    # Evaluate with subsampling
    n_samples = 3
    scores = classification_task.evaluate(
        prompts, 
        classifier_predictor,
        n_samples=n_samples,
        subsample=True
    )
    
    # Verify scores
    assert scores.shape == (1,)  # One score per prompt
    
    # Test with a different random seed to ensure different subsamples
    with pytest.raises(AssertionError, match=r'.*Arrays are not equal.*'):
        # Use a different random seed to force different subsampling
        np.random.seed(42)
        scores1 = classification_task.evaluate(
            prompts, 
            classifier_predictor,
            n_samples=n_samples,
            subsample=True
        )
        
        np.random.seed(43)
        scores2 = classification_task.evaluate(
            prompts, 
            classifier_predictor,
            n_samples=n_samples,
            subsample=True
        )
        
        # This should fail because the subsamples should be different
        np.testing.assert_array_equal(scores1, scores2)


def test_task_evaluate_with_return_seq(classification_task, classifier_predictor):
    """Test the evaluate method with return_seq=True."""
    prompts = ["Classify sentiment:"]
    
    # Evaluate with return_seq=True
    scores, seqs = classification_task.evaluate(
        prompts, 
        classifier_predictor, 
        n_samples=2,
        subsample=True,
        return_seq=True
    )
    
    # Verify scores and sequences
    assert scores.shape == (1,)  # One score per prompt
    assert len(seqs) == 1  # One list of sequences per prompt
    assert len(seqs[0]) == 2  # Two sequences per prompt (n_samples=2)
    
    # Check that sequences contain input text
    for seq in seqs[0]:
        assert any(sample_text in seq for sample_text in classification_task.xs)


def test_task_evaluate_with_system_prompts(classification_task, classifier_predictor, mock_llm_for_classification):
    """Test the evaluate method with system prompts."""
    # Add responses for system prompts
    mock_llm_for_classification.predetermined_responses.update({
        ("Classify sentiment: This product is amazing!", "Be concise"): "positive",
        ("Classify sentiment: I'm disappointed with this purchase.", "Be concise"): "negative",
        ("Classify sentiment: The quality is average, nothing special.", "Be concise"): "neutral",
        ("Classify sentiment: Worst product ever, avoid at all costs!", "Be concise"): "negative",
        ("Classify sentiment: Decent product, does what it's supposed to.", "Be concise"): "positive"
    })
    
    prompts = ["Classify sentiment:"]
    system_prompts = ["Be concise"]
    
    # Evaluate with system prompts
    scores = classification_task.evaluate(
        prompts, 
        classifier_predictor,
        system_prompts=system_prompts
    )
    
    # Verify scores
    assert scores.shape == (1,)
    
    # Verify that system prompts were passed through to the LLM
    assert any(
        call['system_prompts'] == system_prompts
        for call in mock_llm_for_classification.call_history
    )