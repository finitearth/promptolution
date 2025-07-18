import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from promptolution.tasks import ClassificationTask


def test_classification_task_initialization(mock_df):
    """Test that ClassificationTask initializes correctly."""
    task = ClassificationTask(df=mock_df, task_description="Sentiment classification task", x_column="x", y_column="y")

    assert task.task_description == "Sentiment classification task"
    assert len(task.classes) == 3
    assert set(task.classes) == set(["positive", "neutral", "negative"])
    assert len(task.xs) == 3
    assert len(task.ys) == 3
    assert task.metric == accuracy_score


def test_task_evaluate(mock_classification_task_with_subsampling, mock_predictor):
    """Test the evaluate method of ClassificationTask."""
    prompts = ["Classify sentiment:"]
    scores = mock_classification_task_with_subsampling.evaluate(prompts, mock_predictor)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)
    assert 0 <= scores[0] <= 1

    prompts = ["Classify sentiment:", "Rate the text:"]
    scores = mock_classification_task_with_subsampling.evaluate(prompts, mock_predictor)

    assert scores.shape == (2,)
    assert all(0 <= score <= 1 for score in scores)


def test_task_evaluate_with_subsampling(mock_classification_task_with_subsampling, mock_predictor):
    """Test the evaluate method with subsampling."""
    prompts = ["Classify sentiment:"]

    scores = mock_classification_task_with_subsampling.evaluate(
        prompts,
        mock_predictor,
    )

    assert scores.shape == (1,)

    with pytest.raises(AssertionError, match=r".*Arrays are not equal.*"):
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

        np.testing.assert_array_equal(scores1, scores2)


def test_task_evaluate_with_return_seq(mock_classification_task_with_subsampling, mock_predictor):
    """Test the evaluate method with return_seq=True."""
    prompts = ["Classify sentiment:"]

    scores, seqs = mock_classification_task_with_subsampling.evaluate(prompts, mock_predictor, return_seq=True)

    assert scores.shape == (1,)
    assert len(seqs) == 1

    for seq in seqs[0]:
        assert any(sample_text in seq for sample_text in mock_classification_task_with_subsampling.xs)


def test_task_evaluate_with_system_prompts(
    mock_classification_task_with_subsampling, mock_predictor, mock_downstream_llm
):
    """Test the evaluate method with system prompts."""

    prompts = ["Classify sentiment:"]
    system_prompts = ["Be concise"]

    scores = mock_classification_task_with_subsampling.evaluate(
        prompts, mock_predictor, system_prompts=system_prompts, return_agg_scores=True
    )

    assert scores.shape == (1,)
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

    task.increment_block_idx()
    assert task.block_idx == 1

    task.reset_block_idx()
    assert task.block_idx == 0


def test_classification_task_evaluate_random_block(mock_df, mock_predictor):
    """Test the evaluate method with 'random_block' subsampling for ClassificationTask."""
    task = ClassificationTask(
        df=mock_df,
        task_description="Sentiment classification",
        x_column="x",
        y_column="y",
        n_subsamples=1,
        eval_strategy="random_block",
        seed=42,
    )
    prompts = ["Classify sentiment:"]

    evaluated_x_sets = []
    for _ in range(5):
        mock_predictor.call_history = []
        task.evaluate(prompts, mock_predictor)
        if mock_predictor.call_history:
            evaluated_x_sets.append(tuple(mock_predictor.call_history[0]["preds"]))
        else:
            evaluated_x_sets.append(tuple())

    assert len(set(evaluated_x_sets)) > 1, "Should select different random blocks across evaluations"


def test_classification_task_evaluate_sequential_block(mock_df, mock_predictor):
    """Test the evaluate method with 'sequential_block' subsampling for ClassificationTask."""
    task = ClassificationTask(
        df=mock_df,
        task_description="Sentiment classification",
        x_column="x",
        y_column="y",
        n_subsamples=1,
        eval_strategy="sequential_block",
        seed=42,
    )
    prompts = ["Classify sentiment:"]

    task.reset_block_idx()
    assert task.block_idx == 0

    expected_x_sequence = [
        "This review is not negative, so my answer is <final_answer>positive</final_answer>",
        "This review is not positive, so my answer is <final_answer>negative</final_answer>",
        "This review is neither positive nor negative, so my answer is <final_answer>neutral</final_answer>",
    ]

    for i in range(task.n_blocks):
        mock_predictor.call_history = []
        task.evaluate(prompts, mock_predictor)

        assert len(mock_predictor.call_history) == 1
        assert mock_predictor.call_history[0]["preds"][0] == expected_x_sequence[i]

        task.increment_block_idx()
        if i < task.n_blocks - 1:
            assert task.block_idx == i + 1
            assert task.block_idx == 0

    task_full_strategy = ClassificationTask(df=mock_df, x_column="x", y_column="y", eval_strategy="full")
    with pytest.raises(ValueError, match="Block increment is only valid for block subsampling."):
        task_full_strategy.increment_block_idx()
    with pytest.raises(ValueError, match="Block reset is only valid for block subsampling."):
        task_full_strategy.reset_block_idx()
