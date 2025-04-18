# tests/test_optimizers/test_opro.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from promptolution.optimizers.opro import Opro
from promptolution.config import ExperimentConfig
from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_task import MockTask
from tests.mocks.mock_predictor import MockPredictor


@pytest.fixture
def meta_llm_for_opro():
    """Fixture providing a MockLLM for OPRO with appropriate responses."""
    llm = MockLLM()
    
    # Have the LLM return responses with <prompt> tags
    def get_response_with_tags(prompts, system_prompts=None):
        return ["<prompt>Improved classification prompt: Identify the sentiment as positive, neutral, or negative.</prompt>"]
    
    # Instead of overriding _get_response, set up the mock to use get_response
    llm.get_response = get_response_with_tags
    
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
def mock_task_for_opro():
    """Fixture providing a MockTask configured for OPRO testing."""
    task = MockTask(predetermined_scores=[0.7, 0.8, 0.75])
    
    # For OPRO's _sample_examples method
    task.xs = np.array([
        "This is amazing!",
        "I'm disappointed with this.",
        "The quality is average."
    ])
    task.ys = np.array(["positive", "negative", "neutral"])
    
    return task


def test_opro_initialization(meta_llm_for_opro, initial_prompts, mock_task_for_opro):
    """Test that OPRO initializes correctly."""
    optimizer = Opro(
        predictor=MockPredictor(),
        task=mock_task_for_opro,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=meta_llm_for_opro,
        max_num_instructions=10,
        num_instructions_per_step=4,
        num_few_shots=2
    )
    
    # Verify initialization
    assert optimizer.meta_llm == meta_llm_for_opro
    assert optimizer.meta_prompt_template == "<instructions>\n\n<examples>"
    assert optimizer.max_num_instructions == 10
    assert optimizer.num_instructions_per_step == 4
    assert optimizer.num_few_shots == 2
    assert optimizer.prompts == initial_prompts


def test_opro_sample_examples(meta_llm_for_opro, initial_prompts, mock_task_for_opro):
    """Test the _sample_examples method."""
    optimizer = Opro(
        predictor=MockPredictor(),
        task=mock_task_for_opro,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=meta_llm_for_opro,
        num_few_shots=2
    )
    
    # Sample examples
    examples = optimizer._sample_examples()
    
    # Verify that examples were sampled
    assert isinstance(examples, str)
    assert "Input:" in examples
    assert "Output:" in examples
    
    # Should have 2 examples (as set in num_few_shots)
    assert examples.count("Input:") == 2
    assert examples.count("Output:") == 2


def test_opro_format_instructions(meta_llm_for_opro, initial_prompts, mock_task_for_opro):
    """Test the _format_instructions method."""
    optimizer = Opro(
        predictor=MockPredictor(),
        task=mock_task_for_opro,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=meta_llm_for_opro
    )
    
    # Set scores for testing
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.7, 0.9, 0.5]
    
    # Format instructions
    instructions = optimizer._format_instructions()
    
    # Verify that instructions were formatted
    assert isinstance(instructions, str)
    assert "text:" in instructions
    assert "score:" in instructions
    
    # Should have formatted all prompts
    assert instructions.count("text:") == 3
    assert instructions.count("score:") == 3


def test_opro_add_prompt_and_score(meta_llm_for_opro, initial_prompts, mock_task_for_opro):
    """Test the _add_prompt_and_score method."""
    optimizer = Opro(
        predictor=MockPredictor(),
        task=mock_task_for_opro,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=meta_llm_for_opro,
        max_num_instructions=2  # Small value to test truncation
    )
    
    # Set initial state
    optimizer.prompts = ["Prompt 1", "Prompt 2"]
    optimizer.scores = [0.7, 0.6]
    
    # Add a new prompt with higher score
    optimizer._add_prompt_and_score("Prompt 3", 0.8)
    
    # Verify that the prompt was added and the list was truncated
    assert len(optimizer.prompts) == 2
    assert "Prompt 3" in optimizer.prompts
    assert "Prompt 1" in optimizer.prompts
    assert "Prompt 2" not in optimizer.prompts  # Should be removed as lowest score
    
    # Verify that adding a duplicate prompt is ignored
    optimizer._add_prompt_and_score("Prompt 3", 0.9)
    assert len(optimizer.prompts) == 2
    assert optimizer.scores == [0.7, 0.8]  # Score should not be updated


def test_opro_pre_optimization_loop(meta_llm_for_opro, initial_prompts, mock_task_for_opro):
    """Test the _pre_optimization_loop method."""
    optimizer = Opro(
        predictor=MockPredictor(),
        task=mock_task_for_opro,
        initial_prompts=initial_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=meta_llm_for_opro
    )
    
    # Run pre-optimization loop
    optimizer._pre_optimization_loop()
    
    # Verify that scores were evaluated
    assert hasattr(optimizer, 'scores')
    assert len(optimizer.scores) == len(initial_prompts)
    
    # Verify that meta_prompt was created
    assert hasattr(optimizer, 'meta_prompt')
    assert isinstance(optimizer.meta_prompt, str)
    assert "<instructions>" not in optimizer.meta_prompt  # Should be replaced
    assert "<examples>" not in optimizer.meta_prompt  # Should be replaced

def test_opro_step(meta_llm_for_opro, initial_prompts, mock_task_for_opro):
    """Test the _step method."""
    # Configure mock_task to return predetermined scores for any prompt
    def evaluate_mock(prompts, predictor, **kwargs):
        if isinstance(prompts, str):
            if "New unique prompt" in prompts:
                return np.array([0.9])  # Even higher score for our special prompt
            return np.array([0.85])  # Higher score for new prompts
        return np.array([0.7] * len(prompts))
    
    mock_task_for_opro.evaluate = evaluate_mock
    
    # Create a response function that returns unique, identifiable prompts
    def get_unique_response(prompts, system_prompts=None):
        return ["<prompt>New unique prompt #12345: Determine sentiment clearly.</prompt>"]
    
    # Replace the get_response method
    meta_llm_for_opro.get_response = get_unique_response
    
    # Create a smaller initial prompt set to avoid issues
    smaller_initial_set = ["Prompt 1", "Prompt 2"]
    
    optimizer = Opro(
        predictor=MockPredictor(),
        task=mock_task_for_opro,
        initial_prompts=smaller_initial_set,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=meta_llm_for_opro,
        num_instructions_per_step=1  # Generate 1 prompt per step for simplicity
    )
    
    # Set up initial state
    optimizer.prompts = smaller_initial_set
    optimizer.scores = [0.7, 0.65]
    optimizer.meta_prompt = "Meta prompt with instructions and examples"
    
    # Run step
    new_prompts = optimizer._step()
    
    # Verify that new prompts were generated
    assert len(new_prompts) >= len(smaller_initial_set)
    
    # At least one prompt should contain our unique identifier
    assert any("New unique prompt" in prompt for prompt in new_prompts)

def test_opro_optimize(meta_llm_for_opro, initial_prompts, mock_task_for_opro):
    """Test the optimize method."""
    # Configure mock_task to return predetermined scores
    def evaluate_mock(prompts, predictor, **kwargs):
        if isinstance(prompts, str):
            # Return a higher score for a specific new prompt format
            if "New optimized prompt" in prompts:
                return np.array([0.9])
            return np.array([0.85])  # Higher score for other new prompts
        return np.array([0.7] * len(prompts))
    
    mock_task_for_opro.evaluate = evaluate_mock
    
    # Ensure the meta_llm returns a distinctly different prompt
    def get_unique_response(prompts, system_prompts=None):
        # Return a clearly new prompt that won't be in initial_prompts
        return ["<prompt>New optimized prompt: Analyze the text and classify sentiment.</prompt>"]
    
    meta_llm_for_opro.get_response = get_unique_response
    
    # Create OPRO with a smaller subset of initial prompts to avoid test issues
    smaller_initial_set = initial_prompts[:2]  # Just take the first two prompts
    
    optimizer = Opro(
        predictor=MockPredictor(),
        task=mock_task_for_opro,
        initial_prompts=smaller_initial_set,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=meta_llm_for_opro,
        num_instructions_per_step=1  # Generate 1 prompt per step for simplicity
    )
    
    # Run optimization for 2 steps
    optimized_prompts = optimizer.optimize(2)
    
    # Verify that optimization completed and returned prompts
    assert len(optimized_prompts) >= len(smaller_initial_set)
    
    # Check if any new prompts were added (instead of comparing the entire lists)
    assert any("New optimized prompt" in prompt for prompt in optimized_prompts)