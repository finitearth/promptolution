import numpy as np


def test_judge_task_initialization(mock_judge_task_with_y, mock_judge_llm):
    """Test that JudgeTask initializes correctly with ground truth."""
    assert mock_judge_task_with_y.task_description == "Evaluate sentiment prediction quality."
    assert mock_judge_task_with_y.x_column == "x"
    assert mock_judge_task_with_y.y_column == "y"
    assert mock_judge_task_with_y.judge_llm == mock_judge_llm
    assert mock_judge_task_with_y.has_y is True
    assert len(mock_judge_task_with_y.xs) == len(mock_judge_task_with_y.df)
    assert len(mock_judge_task_with_y.ys) == len(mock_judge_task_with_y.df)


def test_judge_task_initialization_no_y(mock_judge_task_no_y, mock_judge_llm):
    """Test JudgeTask initialization when no y_column is provided."""
    assert mock_judge_task_no_y.y_column is None
    assert mock_judge_task_no_y.has_y is False
    assert len(mock_judge_task_no_y.xs) == len(mock_judge_task_no_y.df)
    assert np.all(mock_judge_task_no_y.ys == None)  # noqa: E711


def test_judge_task_construct_judge_prompt_with_ground_truth(mock_judge_task_with_y):
    """Test _construct_judge_prompt generates correct prompt when ground truth is available."""
    x_val = "This movie was great!"
    pred_val = "positive"
    y_val = "positive"
    prompt = mock_judge_task_with_y._construct_judge_prompt(x_val, pred_val, y_val)

    assert mock_judge_task_with_y.task_description in prompt
    assert f"Input:\n{x_val}" in prompt
    assert f"Ground Truth:\n{y_val}" in prompt
    assert f"Prediction:\n{pred_val}" in prompt
    assert "Response:" not in prompt  # Should not use 'Response' if Ground Truth is present
    assert "<final_score>" in prompt  # Ensure the tag is there


def test_judge_task_construct_judge_prompt_without_ground_truth(mock_judge_task_no_y):
    """Test _construct_judge_prompt generates correct prompt when no ground truth."""
    x_val = "Tell me a joke."
    pred_val = "Why did the scarecrow win an award? Because he was outstanding in his field!"
    prompt = mock_judge_task_no_y._construct_judge_prompt(x_val, pred_val, None)  # Pass None for y

    assert mock_judge_task_no_y.task_description in prompt
    assert f"Input:\n{x_val}" in prompt
    assert pred_val in prompt
    assert "<final_score>" in prompt


def test_judge_task_calculate_score_successful_parse(mock_judge_task_with_y, mock_judge_llm):
    """Test _calculate_score correctly parses a valid score from judge LLM response."""
    # Ensure mock_judge_llm provides a specific response for this direct test
    score = mock_judge_task_with_y._calculate_score(x="any", y="any", pred="any")
    assert score == 5.0


def test_judge_task_evaluate_with_ground_truth(mock_judge_task_with_y, mock_predictor, mock_judge_llm):
    """Test the evaluate method of JudgeTask with ground truth and full evaluation."""
    prompts = ["Rate the sentiment:"]

    # Reset call counts on mocks for clean test
    mock_predictor.call_history = []
    mock_judge_llm.call_history = []

    scores_per_datapoint = mock_judge_task_with_y.evaluate(prompts, mock_predictor, return_agg_scores=False)

    # Expect one list of scores (for one prompt) with 5 values (for 5 datapoints)
    assert scores_per_datapoint.shape == (len(prompts), len(mock_judge_task_with_y.xs))

    # Expected scores based on mock_predictor and mock_judge_llm's responses cycle
    expected_scores = [5.0, -5.0, 0.0]  # Based on mock_judge_llm's predetermined_responses cycle
    np.testing.assert_allclose(scores_per_datapoint[0], expected_scores)

    # Verify how many times predictor and judge_llm were called
    assert len(mock_predictor.call_history) == 1  # Predictor called once for the batch
    # Judge LLM is called once for each evaluation (each x, y, pred combination)
    assert len(mock_judge_llm.call_history) == len(mock_judge_task_with_y.xs)

    mock_predictor.call_history = []
    mock_judge_llm.call_history = []

    aggregated_scores = mock_judge_task_with_y.evaluate(prompts, mock_predictor, return_agg_scores=True)
    assert aggregated_scores.shape == (len(prompts),)
    np.testing.assert_allclose(aggregated_scores[0], np.mean(expected_scores))


def test_judge_task_evaluate_no_ground_truth(mock_judge_task_no_y, mock_predictor, mock_judge_llm):
    """Test the evaluate method of JudgeTask without a y_column (no ground truth)."""
    prompts = ["Tell a funny joke:"]

    # # Configure mock_predictor to return specific outputs
    # mock_predictor.set_custom_responses([
    #     "A perfect joke!",  # Will map to judge score 5.0
    #     "An awful joke."    # Will map to judge score -5.0
    # ])

    # Reset call counts
    mock_predictor.call_history = []
    mock_judge_llm.call_history = []

    # mock_judge_task_no_y uses eval_strategy="subsample" with n_subsamples=2
    scores_per_datapoint = mock_judge_task_no_y.evaluate(prompts, mock_predictor, return_agg_scores=False)

    assert scores_per_datapoint.shape == (len(prompts), mock_judge_task_no_y.n_subsamples)

    # Expected scores from mock_judge_llm's responses for the first two subsamples
    expected_scores = [5.0, -5.0]
    np.testing.assert_allclose(scores_per_datapoint[0], expected_scores)

    assert len(mock_predictor.call_history) == 1  # Predictor called once for the batch (of 2 subsamples)
    assert len(mock_judge_llm.call_history) == mock_judge_task_no_y.n_subsamples  # Judge called once per subsample


def test_judge_task_evaluate_with_return_seq(mock_judge_task_with_y, mock_predictor):
    """Test the evaluate method with return_seq=True for JudgeTask."""
    prompts = ["Evaluate this text:"]
    # mock_predictor.set_custom_responses(["Prediction A", "Prediction B", "Prediction C"])

    scores, seqs = mock_judge_task_with_y.evaluate(prompts, mock_predictor, return_seq=True)

    assert scores.shape == (1,)
    assert len(seqs) == 1
    assert len(seqs[0]) == len(mock_judge_task_with_y.xs)
