from unittest.mock import patch

import numpy as np

from promptolution.optimizers import OPRO


def test_opro_initialization(mock_meta_llm, initial_prompts, mock_task, mock_predictor):
    """Test that OPRO initializes correctly."""
    optimizer = OPRO(
        predictor=mock_predictor,
        task=mock_task,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=mock_meta_llm,
        max_num_instructions=10,
        num_instructions_per_step=4,
        num_few_shots=2,
    )

    # Verify only essential properties
    assert optimizer.meta_prompt_template == "<instructions>\n\n<examples>"
    assert optimizer.max_num_instructions == 10
    assert optimizer.num_instructions_per_step == 4
    assert optimizer.num_few_shots == 2
    assert optimizer.prompts == initial_prompts


def test_opro_sample_examples(mock_meta_llm, initial_prompts, mock_task, mock_predictor):
    """Test the _sample_examples method."""
    optimizer = OPRO(
        predictor=mock_predictor,
        task=mock_task,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=mock_meta_llm,
        num_few_shots=2,
    )

    # Control randomness
    with patch("numpy.random.choice") as mock_choice:
        # Return first 2 indices
        mock_choice.return_value = np.array([0, 1])

        # Sample examples
        examples = optimizer._sample_examples()

    # Verify that examples were created
    assert isinstance(examples, str)
    assert "Input:" in examples
    assert "Output:" in examples


def test_opro_format_instructions(mock_meta_llm, initial_prompts, mock_task, mock_predictor):
    """Test the _format_instructions method."""
    optimizer = OPRO(
        predictor=mock_predictor,
        task=mock_task,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=mock_meta_llm,
    )

    # Set scores for testing
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.7, 0.9, 0.5, 0.8, 0.6]

    # Format instructions
    instructions = optimizer._format_instructions()

    # Verify instructions were created
    assert isinstance(instructions, str)
    assert "text:" in instructions
    assert "score:" in instructions


def test_opro_pre_optimization_loop(mock_meta_llm, initial_prompts, mock_task, mock_predictor):
    """Test the _pre_optimization_loop method."""
    optimizer = OPRO(
        predictor=mock_predictor,
        task=mock_task,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=mock_meta_llm,
    )

    # Control randomness
    with patch("numpy.random.choice") as mock_choice:
        # Return first 2 indices
        mock_choice.return_value = np.array([0, 1])

        # Run pre-optimization loop
        optimizer._pre_optimization_loop()

    # Verify that meta_prompt was created
    assert hasattr(optimizer, "meta_prompt")
    assert isinstance(optimizer.meta_prompt, str)


def test_opro_step(mock_meta_llm, initial_prompts, mock_task, mock_predictor):
    """Test the _step method."""
    optimizer = OPRO(
        predictor=mock_predictor,
        task=mock_task,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>Please new Prompt! 🏄🏻‍♀️<examples>",
        meta_llm=mock_meta_llm,
        num_instructions_per_step=1,
    )

    # Set up initial state
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.7, 0.6, 0.5, 0.8]
    optimizer.meta_prompt = "Meta prompt with instructions and examples"

    # Control randomness
    with patch("numpy.random.randint") as mock_randint:
        mock_randint.return_value = 42  # Fixed seed
        new_prompts = optimizer._step()

    # Verify prompts were returned
    assert isinstance(new_prompts, list)
