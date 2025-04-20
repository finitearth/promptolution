import pytest

from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.optimizers.evoprompt_de import EvoPromptDE
from promptolution.optimizers.opro import Opro
from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_task import MockTask
from tests.mocks.mock_predictor import MockPredictor


@pytest.fixture
def meta_llm():
    """Fixture providing a common MockLLM for all optimizers."""
    llm = MockLLM()
    
    # Set up response generation for meta prompts
    def get_response_for_meta(prompts, system_prompts=None):
        return [f"<prompt>Optimized prompt {i}</prompt>" for i in range(len(prompts))]
    
    llm.get_response = get_response_for_meta
    
    return llm


@pytest.fixture
def initial_prompts():
    """Fixture providing a set of initial prompts for testing."""
    return [
        "Classify the following text as positive or negative.",
        "Determine if the sentiment of the text is positive or negative.",
        "Is the following text positive or negative?",
        "Analyze the sentiment of this text and categorize as positive or negative.",
        "Evaluate whether the sentiment expressed is positive or negative.",
    ]


@pytest.fixture
def mock_score_improving_task():
    """Fixture providing a MockTask with scores that improve over iterations."""
    iteration_count = [0]  # Use a list to maintain state across calls
    
    def score_function(prompt):
        # Return higher scores for later iterations
        if "Optimized prompt" in prompt:
            # Extract prompt number
            try:
                num = int(prompt.split("Optimized prompt")[1].strip())
                return min(0.95, 0.7 + 0.05 * num)  # Increase score with prompt number
            except (ValueError, IndexError):
                pass
            
            # For any optimized prompt without a number
            return 0.85
            
        # Base score increases slightly with each iteration
        iteration_count[0] += 1
        return min(0.8, 0.6 + 0.02 * iteration_count[0])
    
    return MockTask(predetermined_scores=score_function)

def test_compare_optimizers(meta_llm, initial_prompts, mock_score_improving_task):
    """Integration test comparing the three optimizers."""
    # Create predictors and task
    predictor = MockPredictor()
    
    # Make a copy of initial_prompts to avoid cross-test contamination
    original_prompts = initial_prompts.copy()
    
    # Create optimizers
    ga_optimizer = EvoPromptGA(
        predictor=predictor,
        task=mock_score_improving_task,
        initial_prompts=original_prompts,
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=meta_llm,
        selection_mode="random",
    )
    
    de_optimizer = EvoPromptDE(
        predictor=predictor,
        task=mock_score_improving_task,
        initial_prompts=original_prompts,
        prompt_template="Create a new prompt from: <prompt0>, <prompt1>, <prompt2>, <prompt3>",
        meta_llm=meta_llm,
        donor_random=False,
    )
    
    opro_optimizer = Opro(
        predictor=predictor,
        task=mock_score_improving_task,
        initial_prompts=original_prompts,
        prompt_template="<instructions>\n\n<examples>",
        meta_llm=meta_llm,
        num_instructions_per_step=2,
    )
    
    # Run optimization for each optimizer
    ga_prompts = ga_optimizer.optimize(2)
    de_prompts = de_optimizer.optimize(2)
    opro_prompts = opro_optimizer.optimize(2)
    
    # Verify that all optimizers completed and returned prompts
    assert len(ga_prompts) == len(initial_prompts)
    assert len(de_prompts) == len(initial_prompts)
    assert len(opro_prompts) >= len(initial_prompts)
    
    # Verify that the optimized prompts are different from the initial prompts
    assert ga_prompts != initial_prompts
    assert de_prompts != initial_prompts
    print(f"OPRO Prompts: {opro_prompts}")
    print("original_prompts: ", initial_prompts)
    assert any(prompt not in initial_prompts for prompt in opro_prompts)
    
    # Evaluate the final best prompt from each optimizer to compare
    ga_score = mock_score_improving_task.evaluate([ga_prompts[0]], predictor)[0]
    de_score = mock_score_improving_task.evaluate([de_prompts[0]], predictor)[0]
    opro_score = mock_score_improving_task.evaluate([opro_prompts[0]], predictor)[0]
    
    # All optimizers should have improved the prompts
    assert ga_score > 0.6
    assert de_score > 0.6
    assert opro_score > 0.6