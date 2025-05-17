from unittest.mock import patch

from promptolution import EvoPromptDE


def test_evoprompt_de_initialization(mock_meta_llm, initial_prompts, mock_task, mock_predictor):
    """Test that EvoPromptDE initializes correctly."""
    optimizer = EvoPromptDE(
        predictor=mock_predictor,
        task=mock_task,
        initial_prompts=initial_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=mock_meta_llm,
        donor_random=False,
    )

    # Only verify the essential properties
    assert optimizer.prompt_template == "Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>"
    assert not optimizer.donor_random
    assert optimizer.prompts == initial_prompts


def test_evoprompt_de_pre_optimization_loop(mock_meta_llm, initial_prompts, mock_task, mock_predictor):
    """Test the _pre_optimization_loop method."""
    optimizer = EvoPromptDE(
        predictor=mock_predictor,
        task=mock_task,
        initial_prompts=initial_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=mock_meta_llm,
    )

    # Call pre-optimization loop
    optimizer._pre_optimization_loop()

    # Simply verify that scores were created and have the right length
    assert hasattr(optimizer, "scores")
    assert len(optimizer.scores) == len(initial_prompts)


def test_evoprompt_de_step(mock_meta_llm, initial_prompts, mock_task, mock_predictor):
    """Test the _step method."""
    optimizer = EvoPromptDE(
        predictor=mock_predictor,
        task=mock_task,
        initial_prompts=initial_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=mock_meta_llm,
        donor_random=False,
    )

    # Set up initial state
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.8, 0.7, 0.6, 0.5, 0.4]  # First prompt is best

    # Control randomness
    with patch("numpy.random.choice") as mock_choice:
        # Return the first 3 elements when size=3
        mock_choice.side_effect = lambda arr, size=None, replace=None: arr[:size] if size else arr[0]

        # Run step
        new_prompts = optimizer._step()

    # Just verify we got the right number of prompts back
    assert len(new_prompts) == len(initial_prompts)


def test_evoprompt_de_optimize(mock_meta_llm, initial_prompts, mock_task, mock_predictor):
    """Test the optimize method."""
    optimizer = EvoPromptDE(
        predictor=mock_predictor,
        task=mock_task,
        initial_prompts=initial_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=mock_meta_llm,
    )

    # Control randomness
    with patch("numpy.random.choice") as mock_choice:
        mock_choice.side_effect = lambda arr, size=None, replace=None: arr[:size] if size else arr[0]

        # Run optimization for 2 steps
        optimized_prompts = optimizer.optimize(2)

    # Just verify we got prompts back
    assert len(optimized_prompts) == len(initial_prompts)
