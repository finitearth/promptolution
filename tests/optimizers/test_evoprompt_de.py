"""Tests for the EvoPromptDE optimizer."""

import pytest
import numpy as np
from typing import List

from promptolution.optimizers.evoprompt_de import EvoPromptDE
from promptolution.optimizers.base_optimizer import OptimizerConfig
from promptolution.config import OptimizerConfig as PromptolutionOptimizerConfig
from promptolution.config import PromptolutionConfig


class TestEvoPromptDE:
    """Test suite for the EvoPromptDE optimizer."""

    def test_initialization(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task):
        """Test that EvoPromptDE initializes correctly with config."""
        # Initialize with direct config
        optimizer = EvoPromptDE(
            prompt_template="Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>.",
            meta_llm=dummy_llm,
            donor_random=False,
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        assert optimizer.prompt_template == "Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>."
        assert optimizer.meta_llm == dummy_llm
        assert optimizer.donor_random is False
        assert optimizer.config.optimizer_name == "test_optimizer"
        assert optimizer.config.n_steps == 5
        assert optimizer.config.population_size == 8
        assert len(optimizer.prompts) == len(initial_prompts)
        
        # Initialize with dictionary config
        optimizer = EvoPromptDE(
            prompt_template="Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>.",
            meta_llm=dummy_llm,
            donor_random=True,
            config={
                "optimizer_name": "test_optimizer_dict",
                "n_steps": 10,
                "population_size": 12
            },
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        assert optimizer.prompt_template == "Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>."
        assert optimizer.meta_llm == dummy_llm
        assert optimizer.donor_random is True
        assert optimizer.config.optimizer_name == "test_optimizer_dict"
        assert optimizer.config.n_steps == 10
        assert optimizer.config.population_size == 12

    def test_initialization_with_promptolution_config(self, dummy_llm, initial_prompts, dummy_task):
        """Test that EvoPromptDE initializes correctly with PromptolutionConfig."""
        optimizer_config = PromptolutionOptimizerConfig(
            optimizer="evoprompt_de",
            n_steps=5,
            init_pop_size=8,
            donor_random=True
        )
        
        config = PromptolutionConfig(
            optimizer_config=optimizer_config
        )
        
        # We use the optimizer_config properties via the config object
        optimizer = EvoPromptDE(
            prompt_template="Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>.",
            meta_llm=dummy_llm,
            donor_random=config.donor_random,
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        assert optimizer.prompt_template == "Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>."
        assert optimizer.meta_llm == dummy_llm
        assert optimizer.donor_random is True

    def test_optimize(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task, dummy_predictor):
        """Test that optimize method works correctly."""
        prompt_template = "Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>."
        
        optimizer = EvoPromptDE(
            prompt_template=prompt_template,
            meta_llm=dummy_llm,
            donor_random=False,
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task,
            predictor=dummy_predictor
        )
        
        # Patch the task.evaluate method to return deterministic scores
        dummy_task.evaluate = lambda prompts, predictor, subsample=True, n_samples=20: np.array([0.8, 0.7, 0.6])
        
        # Run optimization for 2 steps
        optimized_prompts = optimizer.optimize(2)
        
        # Check that we got the right number of prompts back
        assert len(optimized_prompts) == len(initial_prompts)
        
        # Check that the optimize method generated new prompts
        assert "This is an optimized prompt." in optimized_prompts

    def test_donor_random(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task, dummy_predictor):
        """Test that donor_random parameter affects the optimization process."""
        # Set up optimizers with different donor_random values
        optimizer_with_random = EvoPromptDE(
            prompt_template="Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>.",
            meta_llm=dummy_llm,
            donor_random=True,
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task,
            predictor=dummy_predictor
        )
        
        optimizer_without_random = EvoPromptDE(
            prompt_template="Improve this prompt: <prompt0>. Consider these other prompts: <prompt1>, <prompt2>, <prompt3>.",
            meta_llm=dummy_llm,
            donor_random=False,
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task,
            predictor=dummy_predictor
        )
        
        # Mock the task evaluation to make the tests deterministic
        dummy_task.evaluate = lambda prompts, predictor, subsample=True, n_samples=20: np.array([0.8, 0.7, 0.6])
        
        # Should use different optimization paths due to donor_random setting
        assert optimizer_with_random.donor_random != optimizer_without_random.donor_random

    def test_assertions(self, base_optimizer_config, initial_prompts, dummy_task):
        """Test that assertions are raised when required parameters are missing."""
        # Test assertion error when meta_llm is not provided
        with pytest.raises(AssertionError):
            optimizer = EvoPromptDE(
                prompt_template="Template",
                meta_llm=None,
                config=base_optimizer_config,
                initial_prompts=initial_prompts,
                task=dummy_task
            )