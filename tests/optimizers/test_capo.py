import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from promptolution.optimizers.capo import CAPO, CAPOPrompt
from tests.fixtures import meta_llm_mock, initial_prompts, mock_task, mock_predictor, df_few_shots


def test_capo_prompt_initialization():
    """Test that CAPOPrompt initializes correctly."""
    instruction = "Classify the sentiment of the text."
    few_shots = ["Example 1: Positive", "Example 2: Negative"]
    prompt = CAPOPrompt(instruction, few_shots)

    # Verify attributes
    assert prompt.instruction_text == instruction
    assert prompt.few_shots == few_shots


def test_capo_prompt_construct_prompt():
    """Test the construct_prompt method of CAPOPrompt."""
    instruction = "Classify the sentiment of the text."
    few_shots = ["Example 1: Positive", "Example 2: Negative"]
    prompt = CAPOPrompt(instruction, few_shots)

    # Get the constructed prompt
    constructed = prompt.construct_prompt()

    # Verify the prompt contains the instruction
    assert instruction in constructed


def test_capo_initialization(meta_llm_mock, mock_predictor, initial_prompts, mock_task, df_few_shots):
    """Test that CAPO initializes correctly."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=meta_llm_mock,
        initial_prompts=initial_prompts,
        df_few_shots=df_few_shots,
        crossovers_per_iter=3,
        upper_shots=4,
    )

    # Verify essential properties
    assert optimizer.crossovers_per_iter == 3
    assert optimizer.upper_shots == 4
    assert isinstance(optimizer.df_few_shots, pd.DataFrame)


def test_capo_initialize_population(meta_llm_mock, mock_predictor, initial_prompts, mock_task, df_few_shots):
    """Test the _initialize_population method."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=meta_llm_mock,
        initial_prompts=initial_prompts,
        df_few_shots=df_few_shots,
    )

    # Mock the _create_few_shot_examples method to simplify
    def mock_create_few_shot_examples(instruction, num_examples):
        return [f"Example {i}" for i in range(num_examples)]

    optimizer._create_few_shot_examples = mock_create_few_shot_examples

    # Control randomness
    with patch("random.randint", return_value=2):
        population = optimizer._initialize_population(initial_prompts)

    # Verify population was created
    assert len(population) == len(initial_prompts)
    assert all(isinstance(p, CAPOPrompt) for p in population)


def test_capo_step(meta_llm_mock, mock_predictor, initial_prompts, mock_task, df_few_shots):
    """Test the _step method."""
    # Use a smaller population size for the test
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=meta_llm_mock,
        initial_prompts=initial_prompts,
        df_few_shots=df_few_shots,
    )

    # Create mock prompt objects
    mock_prompts = [CAPOPrompt("Instruction 1", ["Example 1"]), CAPOPrompt("Instruction 2", ["Example 2"])]
    optimizer.prompt_objects = mock_prompts

    # Mock the internal methods to avoid complexity
    mock_offspring = [CAPOPrompt("Offspring", ["Example"])]
    optimizer._crossover = lambda x: mock_offspring

    mock_mutated = [CAPOPrompt("Mutated", ["Example"])]
    optimizer._mutate = lambda x: mock_mutated

    mock_survivors = [CAPOPrompt("Survivor 1", ["Example"]), CAPOPrompt("Survivor 2", ["Example"])]
    optimizer._do_racing = lambda x, k: mock_survivors

    # Call _step
    result = optimizer._step()

    # Verify results
    assert len(result) == 2  # Should match population_size
    assert all(isinstance(p, str) for p in result)


def test_capo_optimize(meta_llm_mock, mock_predictor, initial_prompts, mock_task, df_few_shots):
    """Test the optimize method."""
    optimizer = CAPO(
        predictor=mock_predictor,
        task=mock_task,
        meta_llm=meta_llm_mock,
        initial_prompts=initial_prompts,
        df_few_shots=df_few_shots,
    )

    # Mock the internal methods to avoid complexity
    optimizer._pre_optimization_loop = MagicMock()

    def mock_step():
        optimizer.prompts = ["Optimized prompt 1", "Optimized prompt 2"]
        return optimizer.prompts

    optimizer._step = mock_step

    # Call optimize
    optimized_prompts = optimizer.optimize(2)

    # Verify results
    assert len(optimized_prompts) == 2
    assert all(isinstance(p, str) for p in optimized_prompts)

    # Verify method calls
    optimizer._pre_optimization_loop.assert_called_once()
