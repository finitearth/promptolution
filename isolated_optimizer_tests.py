"""Isolated tests for optimizer modules."""

import sys
import os
from typing import List, Optional, Any, Dict, Callable, Union
import unittest
from unittest.mock import MagicMock
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Define mock base classes to avoid dependencies
@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""
    optimizer_name: str = ""
    n_steps: int = 10
    population_size: int = 10
    random_seed: int = 42
    log_path: Optional[str] = None
    n_eval_samples: int = 20
    callbacks: List[str] = field(default_factory=list)

class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers."""
    config_class = OptimizerConfig
    
    def __init__(
        self,
        initial_prompts: List[str] = None,
        task = None,
        callbacks: List[Callable] = None,
        predictor = None,
        config: Optional[Union[Dict[str, Any], OptimizerConfig]] = None,
        **kwargs
    ):
        """Initialize the optimizer."""
        # Initialize config
        if config is None:
            config = {}
            
        if isinstance(config, dict):
            # Merge kwargs into config
            for k, v in kwargs.items():
                config[k] = v
            self.config = self.config_class(**config)
        else:
            self.config = config
            # Override config with kwargs
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        
        # Set up optimizer state
        self.prompts = initial_prompts or []
        self.task = task
        self.callbacks = callbacks or []
        self.predictor = predictor
        self.n_eval_samples = kwargs.get('n_eval_samples', self.config.n_eval_samples)
        
        # Set random seed
        np.random.seed(self.config.random_seed)
    
    @abstractmethod
    def optimize(self, n_steps: Optional[int] = None) -> List[str]:
        """Perform the optimization process."""
        raise NotImplementedError
    
    def _on_step_end(self):
        """Call all registered callbacks at the end of each optimization step."""
        continue_optimization = True
        for callback in self.callbacks:
            continue_optimization &= callback.on_step_end(self)  # if any callback returns False, end the optimization
        
        return continue_optimization
    
    def _on_epoch_end(self):
        """Call all registered callbacks at the end of each optimization epoch."""
        continue_optimization = True
        for callback in self.callbacks:
            continue_optimization &= callback.on_epoch_end(self)  # if any callback returns False, end the optimization
        
        return continue_optimization
    
    def _on_train_end(self):
        """Call all registered callbacks at the end of the entire optimization process."""
        for callback in self.callbacks:
            callback.on_train_end(self)

class BaseLLM:
    """Base class for language models."""
    def get_response(self, prompts: List[str]) -> List[str]:
        """Get responses for prompts."""
        raise NotImplementedError

# Import the optimizers being tested
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from promptolution.optimizers.evoprompt_de import EvoPromptDE
from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.optimizers.opro import Opro

# Mock classes for testing
class MockLLM:
    """Mock language model for testing."""
    def __init__(self, responses=None):
        self.responses = responses or [
            "<prompt>Optimized prompt 1</prompt>",
            "<prompt>Optimized prompt 2</prompt>",
            "<prompt>Optimized prompt 3</prompt>"
        ]
    
    def get_response(self, prompts: List[str]) -> List[str]:
        """Return predetermined responses."""
        return self.responses[:len(prompts)]

class MockTask:
    """Mock task for testing."""
    def __init__(self, eval_scores=None):
        self.xs = np.array(["Example 1", "Example 2"])
        self.ys = np.array(["Label 1", "Label 2"])
        self.eval_scores = eval_scores or np.array([0.8, 0.7, 0.6])
    
    def evaluate(self, prompts, predictor, subsample=True, n_samples=20):
        """Return predetermined evaluation scores."""
        if isinstance(prompts, str):
            return np.array([self.eval_scores[0]])
        return self.eval_scores[:len(prompts)]

class MockPredictor:
    """Mock predictor for testing."""
    def __init__(self):
        self.classes = ["positive", "negative"]

class MockCallback:
    """Mock callback for testing."""
    def __init__(self, return_value=True):
        self.on_step_end_called = 0
        self.on_epoch_end_called = 0
        self.on_train_end_called = 0
        self.return_value = return_value
    
    def on_step_end(self, optimizer):
        self.on_step_end_called += 1
        return self.return_value
    
    def on_epoch_end(self, optimizer):
        self.on_epoch_end_called += 1
        return self.return_value
    
    def on_train_end(self, optimizer):
        self.on_train_end_called += 1
        return True

class TestEvoPromptDE(unittest.TestCase):
    """Test the EvoPromptDE optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_prompts = [
            "Classify the following text as positive or negative.",
            "Determine if the sentiment of the text is positive or negative.",
            "Is the following text positive or negative?",
        ]
        self.task = MockTask()
        self.llm = MockLLM()
        self.predictor = MockPredictor()
        self.prompt_template = "Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>."
    
    def test_initialization(self):
        """Test that EvoPromptDE initializes correctly."""
        optimizer = EvoPromptDE(
            prompt_template=self.prompt_template,
            meta_llm=self.llm,
            donor_random=False,
            config={
                "optimizer_name": "test_optimizer",
                "n_steps": 5,
                "population_size": 8,
                "random_seed": 42
            },
            initial_prompts=self.initial_prompts,
            task=self.task,
            predictor=self.predictor
        )
        
        self.assertEqual(optimizer.prompt_template, self.prompt_template)
        self.assertEqual(optimizer.meta_llm, self.llm)
        self.assertFalse(optimizer.donor_random)
        self.assertEqual(optimizer.config.optimizer_name, "test_optimizer")
        self.assertEqual(optimizer.config.n_steps, 5)
        self.assertEqual(optimizer.config.population_size, 8)
        self.assertEqual(len(optimizer.prompts), len(self.initial_prompts))
    
    def test_optimize(self):
        """Test that optimize method works correctly."""
        optimizer = EvoPromptDE(
            prompt_template=self.prompt_template,
            meta_llm=self.llm,
            donor_random=False,
            config={
                "optimizer_name": "test_optimizer",
                "n_steps": 2,
                "population_size": 3,
                "random_seed": 42
            },
            initial_prompts=self.initial_prompts,
            task=self.task,
            predictor=self.predictor
        )
        
        # Test with a single step
        callback = MockCallback()
        optimizer.callbacks = [callback]
        
        optimized_prompts = optimizer.optimize(1)
        
        # Check that we got the right number of prompts back
        self.assertEqual(len(optimized_prompts), len(self.initial_prompts))
        
        # Check that callbacks were called
        self.assertEqual(callback.on_step_end_called, 1)
        self.assertEqual(callback.on_train_end_called, 1)

class TestEvoPromptGA(unittest.TestCase):
    """Test the EvoPromptGA optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_prompts = [
            "Classify the following text as positive or negative.",
            "Determine if the sentiment of the text is positive or negative.",
            "Is the following text positive or negative?",
        ]
        self.task = MockTask()
        self.llm = MockLLM()
        self.predictor = MockPredictor()
        self.prompt_template = "Combine these prompts to create a better one: <prompt1> and <prompt2>."
    
    def test_initialization(self):
        """Test that EvoPromptGA initializes correctly."""
        for mode in ["random", "wheel", "tour"]:
            optimizer = EvoPromptGA(
                prompt_template=self.prompt_template,
                meta_llm=self.llm,
                selection_mode=mode,
                config={
                    "optimizer_name": "test_optimizer",
                    "n_steps": 5,
                    "population_size": 8,
                    "random_seed": 42
                },
                initial_prompts=self.initial_prompts,
                task=self.task,
                predictor=self.predictor
            )
            
            self.assertEqual(optimizer.prompt_template, self.prompt_template)
            self.assertEqual(optimizer.meta_llm, self.llm)
            self.assertEqual(optimizer.selection_mode, mode)
            self.assertEqual(optimizer.config.optimizer_name, "test_optimizer")
            self.assertEqual(optimizer.config.n_steps, 5)
            self.assertEqual(optimizer.config.population_size, 8)
            self.assertEqual(len(optimizer.prompts), len(self.initial_prompts))
    
    def test_optimize(self):
        """Test that optimize method works correctly."""
        optimizer = EvoPromptGA(
            prompt_template=self.prompt_template,
            meta_llm=self.llm,
            selection_mode="random",
            config={
                "optimizer_name": "test_optimizer",
                "n_steps": 2,
                "population_size": 3,
                "random_seed": 42
            },
            initial_prompts=self.initial_prompts,
            task=self.task,
            predictor=self.predictor
        )
        
        # Test with a single step
        callback = MockCallback()
        optimizer.callbacks = [callback]
        
        optimized_prompts = optimizer.optimize(1)
        
        # Check that we got the right number of prompts back
        self.assertEqual(len(optimized_prompts), len(self.initial_prompts))
        
        # Check that callbacks were called
        self.assertEqual(callback.on_step_end_called, 1)

class TestOpro(unittest.TestCase):
    """Test the Opro optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_prompts = [
            "Classify the following text as positive or negative.",
            "Determine if the sentiment of the text is positive or negative.",
            "Is the following text positive or negative?",
        ]
        self.task = MockTask()
        self.llm = MockLLM()
        self.predictor = MockPredictor()
        self.prompt_template = "Custom template <old_instructions> <examples>"
    
    def test_initialization(self):
        """Test that Opro initializes correctly."""
        optimizer = Opro(
            meta_llm=self.llm,
            n_samples=2,
            prompt_template=self.prompt_template,
            config={
                "optimizer_name": "test_optimizer",
                "n_steps": 5,
                "population_size": 8,
                "random_seed": 42
            },
            initial_prompts=self.initial_prompts,
            task=self.task,
            predictor=self.predictor
        )
        
        self.assertEqual(optimizer.meta_llm, self.llm)
        self.assertEqual(optimizer.n_samples, 2)
        self.assertEqual(optimizer.meta_prompt, self.prompt_template)
        self.assertEqual(optimizer.config.optimizer_name, "test_optimizer")
        self.assertEqual(optimizer.config.n_steps, 5)
        self.assertEqual(optimizer.config.population_size, 8)
    
    def test_sample_examples(self):
        """Test that _sample_examples method works correctly."""
        optimizer = Opro(
            meta_llm=self.llm,
            n_samples=2,
            prompt_template=self.prompt_template,
            config={
                "optimizer_name": "test_optimizer",
                "n_steps": 2,
                "population_size": 3,
                "random_seed": 42
            },
            initial_prompts=self.initial_prompts,
            task=self.task,
            predictor=self.predictor
        )
        
        examples = optimizer._sample_examples()
        
        # Should be a string with Input/Output format
        self.assertIsInstance(examples, str)
        self.assertIn("Input:", examples)
        self.assertIn("Output:", examples)
    
    def test_optimize(self):
        """Test that optimize method works correctly."""
        optimizer = Opro(
            meta_llm=self.llm,
            n_samples=2,
            prompt_template=self.prompt_template,
            config={
                "optimizer_name": "test_optimizer",
                "n_steps": 2,
                "population_size": 3,
                "random_seed": 42
            },
            initial_prompts=self.initial_prompts,
            task=self.task,
            predictor=self.predictor
        )
        
        # Test with a single step
        callback = MockCallback()
        optimizer.callbacks = [callback]
        
        # Mock the evaluation to return consistent results
        optimizer.task.evaluate = lambda p, predictor, subsample=True, n_samples=20: np.array([0.8])
        
        optimized_prompts = optimizer.optimize(1)
        
        # Check that we got more prompts back (Opro adds a prompt each step)
        self.assertEqual(len(optimized_prompts), len(self.initial_prompts) + 1)
        
        # Check that callbacks were called
        self.assertEqual(callback.on_step_end_called, 1)

if __name__ == "__main__":
    unittest.main()