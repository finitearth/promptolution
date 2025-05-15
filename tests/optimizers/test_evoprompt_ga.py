from unittest.mock import patch
from promptolution.optimizers.evoprompt_ga import EvoPromptGA


def test_evoprompt_ga_initialization(mock_meta_llm, initial_prompts, mock_task, experiment_config, mock_predictor):
    """Test that EvoPromptGA initializes correctly."""
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=mock_meta_llm,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=mock_task,
        predictor=mock_predictor,
    )

    # Verify only essential properties
    assert optimizer.prompt_template == "Combine these prompts to create a better one: <prompt1> and <prompt2>."
    assert optimizer.selection_mode == "random"
    assert optimizer.prompts == initial_prompts


def test_evoprompt_ga_crossover(mock_meta_llm, initial_prompts, mock_task, experiment_config, mock_predictor):
    """Test the _crossover method."""
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=mock_meta_llm,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=mock_task,
        predictor=mock_predictor,
    )

    # Set up state for testing
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.8, 0.7, 0.6, 0.5, 0.4]

    # Control randomness
    with patch("numpy.random.choice") as mock_choice:
        # Return first 2 elements when size=2
        mock_choice.side_effect = lambda arr, size=None, replace=None, p=None: arr[:size] if size else arr[0]

        # Test with "random" selection mode
        optimizer.selection_mode = "random"
        child_prompts = optimizer._crossover(optimizer.prompts, optimizer.scores)

    # Just verify we got the expected number of child prompts
    assert len(child_prompts) == len(initial_prompts)


def test_evoprompt_ga_step(mock_meta_llm, initial_prompts, mock_task, experiment_config, mock_predictor):
    """Test the _step method."""
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=mock_meta_llm,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=mock_task,
        predictor=mock_predictor,
    )

    # Set up state for testing
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.8, 0.7, 0.6, 0.5, 0.4]

    # Control randomness
    with patch("numpy.random.choice") as mock_choice:
        mock_choice.side_effect = lambda arr, size=None, replace=None, p=None: arr[:size] if size else arr[0]

        # Call the step method
        new_prompts = optimizer._step()

    # Verify we got prompts back
    assert len(new_prompts) == len(initial_prompts)


def test_evoprompt_ga_optimize(mock_meta_llm, initial_prompts, mock_task, experiment_config, mock_predictor):
    """Test the optimize method."""
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=mock_meta_llm,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=mock_task,
        predictor=mock_predictor,
    )

    # Control randomness
    with patch("numpy.random.choice") as mock_choice:
        mock_choice.side_effect = lambda arr, size=None, replace=None, p=None: arr[:size] if size else arr[0]

        # Run optimization for 2 steps
        optimized_prompts = optimizer.optimize(2)

    # Verify we got prompts back
    assert len(optimized_prompts) == len(initial_prompts)
