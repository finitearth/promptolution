"""Tests for the Opro optimizer."""

import pytest
import numpy as np
from typing import List

from promptolution.optimizers.opro import Opro
from promptolution.optimizers.base_optimizer import OptimizerConfig
from promptolution.config import OptimizerConfig as PromptolutionOptimizerConfig
from promptolution.config import PromptolutionConfig
from promptolution.templates import OPRO_TEMPLATE


class TestOpro:
    """Test suite for the Opro optimizer."""

    def test_initialization(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task):
        """Test that Opro initializes correctly with config."""
        # Initialize with direct config
        optimizer = Opro(
            meta_llm=dummy_llm,
            n_samples=2,
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        assert optimizer.meta_llm == dummy_llm
        assert optimizer.n_samples == 2
        assert optimizer.meta_prompt == OPRO_TEMPLATE
        assert optimizer.config.optimizer_name == "test_optimizer"
        assert optimizer.config.n_steps == 5
        assert optimizer.config.population_size == 8
        assert len(optimizer.prompts) == len(initial_prompts)
        
        # Initialize with custom prompt template
        custom_template = "This is a custom template for OPRO: <old_instructions> <examples>"
        optimizer = Opro(
            meta_llm=dummy_llm,
            n_samples=3,
            prompt_template=custom_template,
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        assert optimizer.meta_llm == dummy_llm
        assert optimizer.n_samples == 3
        assert optimizer.meta_prompt == custom_template
        assert optimizer.config.optimizer_name == "test_optimizer"

    def test_initialization_with_promptolution_config(self, dummy_llm, initial_prompts, dummy_task):
        """Test that Opro initializes correctly with PromptolutionConfig."""
        optimizer_config = PromptolutionOptimizerConfig(
            optimizer="opro",
            n_steps=5,
            init_pop_size=8
        )
        
        config = PromptolutionConfig(
            optimizer_config=optimizer_config
        )
        
        # Use the optimizer_config properties via the config object
        optimizer = Opro(
            meta_llm=dummy_llm,
            n_samples=2,
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        assert optimizer.meta_llm == dummy_llm
        assert optimizer.n_samples == 2
        assert optimizer.meta_prompt == OPRO_TEMPLATE

    def test_sample_examples(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task):
        """Test that _sample_examples method works correctly."""
        optimizer = Opro(
            meta_llm=dummy_llm,
            n_samples=2,
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        # Call the sample method
        examples = optimizer._sample_examples()
        
        # Check that we got a string back
        assert isinstance(examples, str)
        
        # Check that the examples contain input/output format
        assert "Input:" in examples
        assert "Output:" in examples

    def test_format_old_instructions(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task):
        """Test that _format_old_instructions method works correctly."""
        optimizer = Opro(
            meta_llm=dummy_llm,
            n_samples=2,
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        # Set up some scores
        optimizer.scores = [0.8, 0.7, 0.6]
        
        # Call the format method
        formatted = optimizer._format_old_instructions()
        
        # Check that we got a string back
        assert isinstance(formatted, str)
        
        # Check that each prompt and score is in the output
        for prompt, score in zip(initial_prompts, optimizer.scores):
            assert prompt in formatted
            assert str(score) in formatted

    def test_optimize(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task, dummy_predictor):
        """Test that optimize method works correctly."""
        optimizer = Opro(
            meta_llm=dummy_llm,
            n_samples=2,
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task,
            predictor=dummy_predictor
        )
        
        # Patch the task.evaluate method to return a deterministic score
        original_evaluate = dummy_task.evaluate
        # First, make the initial evaluation deterministic
        dummy_task.evaluate = lambda p, predictor, subsample=True, n_samples=20: np.array([0.8]) if isinstance(p, str) else np.array([0.8, 0.7, 0.6])
        
        # Run optimization for 2 steps
        optimized_prompts = optimizer.optimize(2)
        
        # Check that we got more prompts back
        assert len(optimized_prompts) > len(initial_prompts)
        
        # Check that the new prompts contain the optimized prompt from our dummy LLM
        assert any("This is an optimized prompt." in p for p in optimized_prompts)

    def test_assertions(self, base_optimizer_config, initial_prompts, dummy_task):
        """Test that assertions are raised when required parameters are missing or invalid."""
        # Test assertion error when n_samples is 0
        with pytest.raises(AssertionError):
            optimizer = Opro(
                meta_llm=dummy_llm,
                n_samples=0,
                config=base_optimizer_config,
                initial_prompts=initial_prompts,
                task=dummy_task
            )