# tests/test_optimizers/test_evoprompt_ga.py
import pytest
import numpy as np
from unittest.mock import patch

from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.config import ExperimentConfig
from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_task import MockTask
from tests.mocks.mock_predictor import MockPredictor


@pytest.fixture
def meta_llm_mock():
    """Fixture providing a MockLLM configured for meta-prompt responses."""
    # Responses for meta prompts that extract the prompt between <prompt> tags
    responses = {
        # Simple replacement pattern for testing
        "Combine these prompts to create a better one: <prompt1> and <prompt2>.": "<prompt>Combined prompt</prompt>",
        "Improve upon these prompts: <prompt1> and <prompt2>.": "<prompt>Improved prompt</prompt>",
    }
    
    # For more complex patterns, we can use a function
    def meta_prompt_response_generator(prompt):
        if "Combine these prompts" in prompt:
            # Extract the prompts from the meta-prompt
            parts = prompt.split("<prompt1>")[1].split("</prompt1>")[0]
            return f"<prompt>Combined: {parts[:20]}...</prompt>"
        elif "Improve upon" in prompt:
            return "<prompt>Improved version of prompt</prompt>"
        else:
            return "<prompt>Default meta-response</prompt>"
    
    # Create a more sophisticated mock that generates responses based on inputs
    llm = MockLLM()
    llm._get_response = lambda prompts, system_prompts: [
        f"<prompt>Meta-generated prompt for input {i}</prompt>" for i, _ in enumerate(prompts)
    ]
    
    return llm


@pytest.fixture
def initial_prompts():
    """Fixture providing a set of initial prompts for testing."""
    return [
        "Classify the following text as positive or negative.",
        "Determine if the sentiment of the text is positive or negative.",
        "Is the following text positive or negative?",
    ]


@pytest.fixture
def experiment_config():
    """Fixture providing a basic experiment configuration."""
    return ExperimentConfig(
        optimizer_name="test_optimizer",
        n_steps=3,
        population_size=3,
        random_seed=42
    )


@pytest.fixture
def mock_task_with_scores():
    """Fixture providing a MockTask with predetermined scoring behavior."""
    # A function that generates scores based on the prompt
    def score_function(prompt):
        # Prefer longer prompts for testing purposes
        return min(0.9, 0.5 + 0.01 * len(prompt))
    
    return MockTask(predetermined_scores=score_function)


def test_evoprompt_ga_initialization(meta_llm_mock, initial_prompts, experiment_config, mock_task_with_scores):
    """Test that EvoPromptGA initializes correctly."""
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=meta_llm_mock,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=mock_task_with_scores,
        predictor=MockPredictor()
    )
    
    # Verify initialization
    assert optimizer.prompt_template == "Combine these prompts to create a better one: <prompt1> and <prompt2>."
    assert optimizer.meta_llm == meta_llm_mock
    assert optimizer.selection_mode == "random"
    assert optimizer.config.optimizer_name == "test_optimizer"
    assert optimizer.config.n_steps == 3
    assert optimizer.config.population_size == 3
    assert optimizer.task == mock_task_with_scores
    assert optimizer.prompts == initial_prompts


def test_evoprompt_ga_pre_optimization_loop(meta_llm_mock, initial_prompts, experiment_config, mock_task_with_scores):
    """Test the _pre_optimization_loop method."""
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=meta_llm_mock,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=mock_task_with_scores,
        predictor=MockPredictor()
    )
    
    # Call _pre_optimization_loop directly
    optimizer._pre_optimization_loop()
    
    # Verify that the task's evaluate method was called
    assert len(mock_task_with_scores.call_history) == 1
    assert mock_task_with_scores.call_history[0]['prompts'] == initial_prompts
    
    # Verify that scores and prompts were set and sorted
    assert hasattr(optimizer, 'scores')
    assert len(optimizer.scores) == len(initial_prompts)
    
    # Check if sorting by score happened
    assert optimizer.scores == sorted(optimizer.scores, reverse=True)


def test_evoprompt_ga_crossover(meta_llm_mock, initial_prompts, experiment_config):
    """Test the _crossover method."""
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=meta_llm_mock,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=MockTask(),
        predictor=MockPredictor()
    )
    
    # Set up scores for testing
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.8, 0.7, 0.6]
    
    # Test random selection mode
    optimizer.selection_mode = "random"
    child_prompts_random = optimizer._crossover(optimizer.prompts, optimizer.scores)
    assert len(child_prompts_random) == len(initial_prompts)
    
    # Test wheel selection mode
    optimizer.selection_mode = "wheel"
    child_prompts_wheel = optimizer._crossover(optimizer.prompts, optimizer.scores)
    assert len(child_prompts_wheel) == len(initial_prompts)
    
    # Test tournament selection mode
    optimizer.selection_mode = "tour"
    child_prompts_tour = optimizer._crossover(optimizer.prompts, optimizer.scores)
    assert len(child_prompts_tour) == len(initial_prompts)


def test_evoprompt_ga_step(meta_llm_mock, initial_prompts, experiment_config, mock_task_with_scores):
    """Test the _step method."""
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=meta_llm_mock,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=mock_task_with_scores,
        predictor=MockPredictor()
    )
    
    # Set up state for testing
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.8, 0.7, 0.6]
    
    # Call the step method
    new_prompts = optimizer._step()
    
    # Verify results
    assert len(new_prompts) == len(initial_prompts)
    
    # Check that task.evaluate was called
    assert len(mock_task_with_scores.call_history) >= 1
    
    # Check that scores were updated
    assert hasattr(optimizer, 'scores')
    assert len(optimizer.scores) == len(initial_prompts)


def test_evoprompt_ga_optimize(meta_llm_mock, initial_prompts, experiment_config, mock_task_with_scores):
    """Test the optimize method."""
    # Create mock callback
    mock_callback = type('MockCallback', (), {
        'on_step_end': lambda self, optimizer: True,
        'on_train_end': lambda self, optimizer: None
    })()
    
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=meta_llm_mock,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=mock_task_with_scores,
        predictor=MockPredictor(),
        callbacks=[mock_callback]
    )
    
    # Run optimization
    optimized_prompts = optimizer.optimize(2)
    
    # Verify results
    assert len(optimized_prompts) == len(initial_prompts)
    
    # Check that the optimization process produced different prompts
    assert optimized_prompts != initial_prompts
    
    # Check that task.evaluate was called multiple times
    assert len(mock_task_with_scores.call_history) >= 2


def test_evoprompt_ga_with_early_stopping(meta_llm_mock, initial_prompts, experiment_config, mock_task_with_scores):
    """Test optimization with early stopping."""
    # Create mock callback that stops after first step
    call_count = 0
    mock_callback = type('MockCallback', (), {
        'on_step_end': lambda self, optimizer: False if call_count > 0 else True,
        'on_train_end': lambda self, optimizer: None
    })()
    
    # Mock the callback's on_step_end to count calls and stop after the first
    def mock_on_step_end(optimizer):
        nonlocal call_count
        call_count += 1
        return call_count <= 1
    
    mock_callback.on_step_end = mock_on_step_end
    
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=meta_llm_mock,
        selection_mode="random",
        config=experiment_config,
        initial_prompts=initial_prompts,
        task=mock_task_with_scores,
        predictor=MockPredictor(),
        callbacks=[mock_callback]
    )
    
    # Run optimization for 5 steps, but it should stop after 1
    optimized_prompts = optimizer.optimize(5)
    
    # Verify results
    assert len(optimized_prompts) == len(initial_prompts)
    
    # Check that the callback was called exactly twice (once for stopping)
    assert call_count == 2