import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from promptolution.tasks import ClassificationTask


def test_classification_task_initialization(mock_df):
    """Test that ClassificationTask initializes correctly."""
    task = ClassificationTask(df=mock_df, task_description="Sentiment classification task", x_column="x", y_column="y")

    # Verify attributes
    assert task.task_description == "Sentiment classification task"
    assert len(task.classes) == 3
    assert set(task.classes) == set(["positive", "neutral", "negative"])
    assert len(task.xs) == 3
    assert len(task.ys) == 3
    assert task.metric == accuracy_score


def test_task_evaluate(mock_classification_task_with_subsampling, mock_predictor):
    """Test the evaluate method of ClassificationTask."""
    # Evaluate with a single prompt
    prompts = ["Classify sentiment:"]
    scores = mock_classification_task_with_subsampling.evaluate(prompts, mock_predictor)

    # Verify scores
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)  # One score per prompt
    assert 0 <= scores[0] <= 1  # Score should be between 0 and 1

    # Evaluate with multiple prompts
    prompts = ["Classify sentiment:", "Rate the text:"]
    scores = mock_classification_task_with_subsampling.evaluate(prompts, mock_predictor)

    # Verify scores for multiple prompts
    assert scores.shape == (2,)  # Two scores, one per prompt
    assert all(0 <= score <= 1 for score in scores)


def test_task_evaluate_with_subsampling(mock_classification_task_with_subsampling, mock_predictor):
    """Test the evaluate method with subsampling."""
    prompts = ["Classify sentiment:"]

    # Evaluate with subsampling
    scores = mock_classification_task_with_subsampling.evaluate(
        prompts,
        mock_predictor,
    )

    # Verify scores
    assert scores.shape == (1,)  # One score per prompt

    # Test with a different random seed to ensure different subsamples
    with pytest.raises(AssertionError, match=r".*Arrays are not equal.*"):
        # Use a different random seed to force different subsampling
        np.random.seed(42)
        scores1 = mock_classification_task_with_subsampling.evaluate(
            prompts,
            mock_predictor,
        )

        np.random.seed(43)
        scores2 = mock_classification_task_with_subsampling.evaluate(
            prompts,
            mock_predictor,
        )

        # This should fail because the subsamples should be different
        np.testing.assert_array_equal(scores1, scores2)


def test_task_evaluate_with_return_seq(mock_classification_task_with_subsampling, mock_predictor):
    """Test the evaluate method with return_seq=True."""
    prompts = ["Classify sentiment:"]

    # Evaluate with return_seq=True
    scores, seqs = mock_classification_task_with_subsampling.evaluate(prompts, mock_predictor, return_seq=True)

    # Verify scores and sequences
    assert scores.shape == (1,)  # One score per prompt
    assert len(seqs) == 1  # One list of sequences per prompt

    # Check that sequences contain input text
    for seq in seqs[0]:
        assert any(sample_text in seq for sample_text in mock_classification_task_with_subsampling.xs)


def test_task_evaluate_with_system_prompts(
    mock_classification_task_with_subsampling, mock_predictor, mock_downstream_llm
):
    """Test the evaluate method with system prompts."""

    prompts = ["Classify sentiment:"]
    system_prompts = ["Be concise"]

    # Evaluate with system prompts
    scores = mock_classification_task_with_subsampling.evaluate(
        prompts, mock_predictor, system_prompts=system_prompts, return_agg_scores=True
    )

    # Verify scores
    assert scores.shape == (1,)

    # Verify that system prompts were passed through to the LLM
    assert any(call["system_prompts"] == system_prompts for call in mock_downstream_llm.call_history)


def test_pop_datapoints(mock_df):
    task = ClassificationTask(
        df=mock_df,
        task_description="Sentiment classification task",
        eval_strategy="sequential_blocks",
    )

    df = task.pop_datapoints(n=1)
    assert len(df) == 1
    assert df["x"].values[0] not in task.xs
    assert df["y"].values[0] not in task.ys


def test_blocks(mock_df):
    task = ClassificationTask(
        df=mock_df, task_description="Sentiment classification task", eval_strategy="sequential_blocks", n_subsamples=1
    )

    # Increment blocks
    task.increment_block_idx()
    assert task.block_idx == 1

    # Reset blocks
    task.reset_block_idx()
    assert task.block_idx == 0
