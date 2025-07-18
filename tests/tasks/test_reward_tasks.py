import numpy as np


def test_reward_task_initialization(mock_reward_task, simple_reward_function):
    """Test that RewardTask initializes correctly."""
    assert mock_reward_task.task_description == "Evaluate text quality"
    assert mock_reward_task.reward_function == simple_reward_function
    assert mock_reward_task.x_column == "x"
    assert not mock_reward_task.has_y
    assert len(mock_reward_task.xs) == len(mock_reward_task.df)
    assert np.all(mock_reward_task.ys == None)  # noqa: E711


def test_reward_task_initialization_no_x_column(mock_reward_task_no_x_column, simple_reward_function):
    """Test RewardTask initialization when a dummy x_column is provided (no semantic input)."""
    assert mock_reward_task_no_x_column.x_column == "dummy_input"
    assert not mock_reward_task_no_x_column.has_y
    assert len(mock_reward_task_no_x_column.xs) == len(mock_reward_task_no_x_column.df)
    assert all(x == "" for x in mock_reward_task_no_x_column.xs)
    assert np.all(mock_reward_task_no_x_column.ys == None)  # noqa: E711


def test_reward_task_evaluate_with_return_seq(mock_reward_task, mock_predictor):
    """Test the evaluate method with return_seq=True for RewardTask."""
    prompts = ["Generate a short text:"]

    scores, seqs = mock_reward_task.evaluate(prompts, mock_predictor, return_seq=True, return_agg_scores=False)

    assert scores.shape == (1, len(mock_reward_task.xs))
    assert len(seqs) == 1
    assert len(seqs[0]) == len(mock_reward_task.xs)

    for seq in seqs[0]:
        assert any(str(x_val) in seq for x_val in mock_reward_task.xs)
