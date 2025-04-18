import pytest
import numpy as np
from unittest.mock import patch

from promptolution.optimizers.evoprompt_de import EvoPromptDE
from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_task import MockTask
from tests.mocks.mock_predictor import MockPredictor


@pytest.fixture
def meta_llm_for_de():
    """Fixture providing a MockLLM for EvoPromptDE with appropriate responses."""
    llm = MockLLM()
    
    # Set up response generation for meta prompts
    def get_response_for_de(prompts, system_prompts=None):
        return [f"<prompt>DE improved prompt {i}</prompt>" for i in range(len(prompts))]
    
    llm.get_response = get_response_for_de
    
    return llm


@pytest.fixture
def initial_prompts():
    """Fixture providing a set of initial prompts for testing."""
    return [
        "Classify the following text as positive or negative.",
        "Determine if the sentiment of the text is positive or negative.",
        "Is the following text positive or negative?",
        "Analyze the sentiment in the following text.",
        "Evaluate the sentiment of the text.",
    ]


@pytest.fixture
def mock_task_for_de():
    """Fixture providing a MockTask for EvoPromptDE."""
    # Create scores that improve for "DE improved" prompts
    def score_function(prompt):
        if "DE improved" in prompt:
            return 0.85  # Higher score for DE-generated prompts
        return 0.7  # Base score for initial prompts
    
    return MockTask(predetermined_scores=score_function)


def test_evoprompt_de_initialization(meta_llm_for_de, initial_prompts, mock_task_for_de):
    """Test that EvoPromptDE initializes correctly."""
    optimizer = EvoPromptDE(
        predictor=MockPredictor(),
        task=mock_task_for_de,
        initial_prompts=initial_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=meta_llm_for_de,
        donor_random=False,
        n_eval_samples=15
    )
    
    # Verify initialization
    assert optimizer.meta_llm == meta_llm_for_de
    assert optimizer.prompt_template == "Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>"
    assert optimizer.donor_random == False
    assert optimizer.n_eval_samples == 15
    assert optimizer.prompts == initial_prompts


def test_evoprompt_de_pre_optimization_loop(meta_llm_for_de, initial_prompts, mock_task_for_de):
    """Test the _pre_optimization_loop method."""
    optimizer = EvoPromptDE(
        predictor=MockPredictor(),
        task=mock_task_for_de,
        initial_prompts=initial_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=meta_llm_for_de
    )
    
    # Call pre-optimization loop
    optimizer._pre_optimization_loop()
    
    # Verify that scores were evaluated
    assert hasattr(optimizer, 'scores')
    assert len(optimizer.scores) == len(initial_prompts)
    
    # Verify that prompts and scores were sorted by score (descending)
    assert all(optimizer.scores[i] >= optimizer.scores[i+1] for i in range(len(optimizer.scores)-1))


def test_evoprompt_de_step_with_donor_random(meta_llm_for_de, initial_prompts, mock_task_for_de):
    """Test the _step method with donor_random=True."""
    optimizer = EvoPromptDE(
        predictor=MockPredictor(),
        task=mock_task_for_de,
        initial_prompts=initial_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=meta_llm_for_de,
        donor_random=True
    )
    
    # Set up initial state
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.7, 0.75, 0.65, 0.8, 0.6] 
    
    # Run step
    new_prompts = optimizer._step()
    
    # Verify that step returned expected prompts
    assert len(new_prompts) == len(initial_prompts)
    
    # Prompts should be sorted by score
    assert all(optimizer.scores[i] >= optimizer.scores[i+1] for i in range(len(optimizer.scores)-1))


def test_evoprompt_de_step_with_best_donor(meta_llm_for_de, initial_prompts, mock_task_for_de):
    """Test the _step method with donor_random=False (using best prompt as donor)."""
    optimizer = EvoPromptDE(
        predictor=MockPredictor(),
        task=mock_task_for_de,
        initial_prompts=initial_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=meta_llm_for_de,
        donor_random=False
    )
    
    # Set up initial state
    optimizer.prompts = initial_prompts
    optimizer.scores = [0.8, 0.7, 0.6, 0.4, 0.3]  # First prompt is best
    
    # Patch np.random.choice to control randomness
    with patch('numpy.random.choice') as mock_choice:
        # Set up mock to return predictable choices
        mock_choice.side_effect = lambda arr, size=None, replace=None: np.array([arr[0], arr[1], arr[2]])
        
        # Run step
        new_prompts = optimizer._step()
    
    # Verify that step returned expected prompts
    assert len(new_prompts) == len(initial_prompts)


def test_evoprompt_de_optimize(meta_llm_for_de, initial_prompts, mock_task_for_de):
    """Test the optimize method."""
    optimizer = EvoPromptDE(
        predictor=MockPredictor(),
        task=mock_task_for_de,
        initial_prompts=initial_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=meta_llm_for_de
    )
    
    # Run optimization for 2 steps
    optimized_prompts = optimizer.optimize(2)
    
    # Verify that optimization completed and returned prompts
    assert len(optimized_prompts) == len(initial_prompts)
    
    # The prompts should have been improved
    assert any("DE improved" in prompt for prompt in optimized_prompts)
    