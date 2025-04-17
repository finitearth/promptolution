"""Standalone test to verify optimizer logic."""

import sys
import os
from typing import List, Optional, Any, Dict, Callable, Union
from dataclasses import dataclass, field
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the optimizer modules directly
from promptolution.optimizers.base_optimizer import OptimizerConfig, BaseOptimizer
from promptolution.optimizers.evoprompt_de import EvoPromptDE
from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.optimizers.opro import Opro

# Create mock classes to avoid dependencies
class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self):
        self.responses = [
            "<prompt>Optimized prompt 1</prompt>",
            "<prompt>Optimized prompt 2</prompt>",
            "<prompt>Optimized prompt 3</prompt>",
        ]
    
    def get_response(self, prompts):
        """Return mock responses."""
        return self.responses[:len(prompts)]

class MockTask:
    """Mock task for testing."""
    
    def __init__(self):
        self.xs = np.array(["Example 1", "Example 2"])
        self.ys = np.array(["Label 1", "Label 2"])
    
    def evaluate(self, prompts, predictor, subsample=True, n_samples=20):
        """Return mock evaluation scores."""
        if isinstance(prompts, str):
            return np.array([0.8])
        else:
            return np.array([0.8, 0.7, 0.6][:len(prompts)])

class MockPredictor:
    """Mock predictor for testing."""
    
    def __init__(self):
        self.classes = ["positive", "negative"]

# Test OptimizerConfig
def test_optimizer_config():
    """Test that OptimizerConfig works correctly."""
    config = OptimizerConfig(
        optimizer_name="test_optimizer",
        n_steps=10,
        population_size=12,
        random_seed=42
    )
    
    assert config.optimizer_name == "test_optimizer"
    assert config.n_steps == 10
    assert config.population_size == 12
    assert config.random_seed == 42

# Test EvoPromptDE
def test_evoprompt_de():
    """Test that EvoPromptDE initializes and runs correctly."""
    initial_prompts = [
        "Classify the following text as positive or negative.",
        "Determine if the sentiment of the text is positive or negative.",
        "Is the following text positive or negative?",
    ]
    
    optimizer = EvoPromptDE(
        prompt_template="Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>.",
        meta_llm=MockLLM(),
        donor_random=False,
        config={
            "optimizer_name": "test_optimizer",
            "n_steps": 2,
            "population_size": 3,
            "random_seed": 42
        },
        initial_prompts=initial_prompts,
        task=MockTask(),
        predictor=MockPredictor()
    )
    
    # Verify initialization
    assert optimizer.prompt_template == "Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>."
    assert optimizer.donor_random is False
    assert optimizer.config.optimizer_name == "test_optimizer"
    assert optimizer.config.n_steps == 2
    assert optimizer.config.population_size == 3
    
    # Run optimization
    optimized_prompts = optimizer.optimize(1)
    
    # Verify optimization produced results
    assert len(optimized_prompts) == len(initial_prompts)
    
    # Test completed successfully
    print("EvoPromptDE test passed")

# Test EvoPromptGA
def test_evoprompt_ga():
    """Test that EvoPromptGA initializes and runs correctly."""
    initial_prompts = [
        "Classify the following text as positive or negative.",
        "Determine if the sentiment of the text is positive or negative.",
        "Is the following text positive or negative?",
    ]
    
    optimizer = EvoPromptGA(
        prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
        meta_llm=MockLLM(),
        selection_mode="random",
        config={
            "optimizer_name": "test_optimizer",
            "n_steps": 2,
            "population_size": 3,
            "random_seed": 42
        },
        initial_prompts=initial_prompts,
        task=MockTask(),
        predictor=MockPredictor()
    )
    
    # Verify initialization
    assert optimizer.prompt_template == "Combine these prompts to create a better one: <prompt1> and <prompt2>."
    assert optimizer.selection_mode == "random"
    assert optimizer.config.optimizer_name == "test_optimizer"
    assert optimizer.config.n_steps == 2
    assert optimizer.config.population_size == 3
    
    # Run optimization
    optimized_prompts = optimizer.optimize(1)
    
    # Verify optimization produced results
    assert len(optimized_prompts) == len(initial_prompts)
    
    # Test completed successfully
    print("EvoPromptGA test passed")

# Test Opro
def test_opro():
    """Test that Opro initializes and runs correctly."""
    initial_prompts = [
        "Classify the following text as positive or negative.",
        "Determine if the sentiment of the text is positive or negative.",
        "Is the following text positive or negative?",
    ]
    
    optimizer = Opro(
        meta_llm=MockLLM(),
        n_samples=2,
        prompt_template="Custom template <old_instructions> <examples>",
        config={
            "optimizer_name": "test_optimizer",
            "n_steps": 2,
            "population_size": 3,
            "random_seed": 42
        },
        initial_prompts=initial_prompts,
        task=MockTask(),
        predictor=MockPredictor()
    )
    
    # Verify initialization
    assert optimizer.n_samples == 2
    assert optimizer.meta_prompt == "Custom template <old_instructions> <examples>"
    assert optimizer.config.optimizer_name == "test_optimizer"
    assert optimizer.config.n_steps == 2
    assert optimizer.config.population_size == 3
    
    # Run optimization
    optimized_prompts = optimizer.optimize(1)
    
    # Verify optimization produced results
    assert len(optimized_prompts) > len(initial_prompts)
    
    # Test completed successfully
    print("Opro test passed")

if __name__ == "__main__":
    # Run all tests manually
    print("Running standalone tests for optimizers...")
    test_optimizer_config()
    test_evoprompt_de()
    test_evoprompt_ga()
    test_opro()
    print("All tests passed!")