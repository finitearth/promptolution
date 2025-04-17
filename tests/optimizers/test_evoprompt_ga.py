"""Tests for the EvoPromptGA optimizer."""

import pytest
import numpy as np
from typing import List

from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.optimizers.base_optimizer import OptimizerConfig
from promptolution.config import OptimizerConfig as PromptolutionOptimizerConfig
from promptolution.config import PromptolutionConfig


class TestEvoPromptGA:
    """Test suite for the EvoPromptGA optimizer."""

    def test_initialization(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task):
        """Test that EvoPromptGA initializes correctly with config."""
        # Initialize with direct config
        optimizer = EvoPromptGA(
            prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
            meta_llm=dummy_llm,
            selection_mode="wheel",
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        assert optimizer.prompt_template == "Combine these prompts to create a better one: <prompt1> and <prompt2>."
        assert optimizer.meta_llm == dummy_llm
        assert optimizer.selection_mode == "wheel"
        assert optimizer.config.optimizer_name == "test_optimizer"
        assert optimizer.config.n_steps == 5
        assert optimizer.config.population_size == 8
        assert len(optimizer.prompts) == len(initial_prompts)
        
        # Initialize with dictionary config
        optimizer = EvoPromptGA(
            prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
            meta_llm=dummy_llm,
            selection_mode="tour",
            config={
                "optimizer_name": "test_optimizer_dict",
                "n_steps": 10,
                "population_size": 12
            },
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        assert optimizer.prompt_template == "Combine these prompts to create a better one: <prompt1> and <prompt2>."
        assert optimizer.meta_llm == dummy_llm
        assert optimizer.selection_mode == "tour"
        assert optimizer.config.optimizer_name == "test_optimizer_dict"
        assert optimizer.config.n_steps == 10
        assert optimizer.config.population_size == 12

    def test_initialization_with_promptolution_config(self, dummy_llm, initial_prompts, dummy_task):
        """Test that EvoPromptGA initializes correctly with PromptolutionConfig."""
        optimizer_config = PromptolutionOptimizerConfig(
            optimizer="evoprompt_ga",
            n_steps=5,
            init_pop_size=8,
            selection_mode="random"
        )
        
        config = PromptolutionConfig(
            optimizer_config=optimizer_config
        )
        
        # We use the optimizer_config properties via the config object
        optimizer = EvoPromptGA(
            prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
            meta_llm=dummy_llm,
            selection_mode=config.selection_mode,
            initial_prompts=initial_prompts,
            task=dummy_task
        )
        
        assert optimizer.prompt_template == "Combine these prompts to create a better one: <prompt1> and <prompt2>."
        assert optimizer.meta_llm == dummy_llm
        assert optimizer.selection_mode == "random"

    def test_optimize(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task, dummy_predictor):
        """Test that optimize method works correctly."""
        prompt_template = "Combine these prompts to create a better one: <prompt1> and <prompt2>."
        
        optimizer = EvoPromptGA(
            prompt_template=prompt_template,
            meta_llm=dummy_llm,
            selection_mode="random",
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
        
        # Check that the prompts were sorted by score (best first)
        assert optimized_prompts[0] != optimized_prompts[-1]

    def test_selection_modes(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task, dummy_predictor):
        """Test that different selection modes can be used."""
        for mode in ["random", "wheel", "tour"]:
            optimizer = EvoPromptGA(
                prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
                meta_llm=dummy_llm,
                selection_mode=mode,
                config=base_optimizer_config,
                initial_prompts=initial_prompts,
                task=dummy_task,
                predictor=dummy_predictor
            )
            
            assert optimizer.selection_mode == mode
            
            # Mock the task evaluation to make the tests deterministic
            dummy_task.evaluate = lambda prompts, predictor, subsample=True, n_samples=20: np.array([0.8, 0.7, 0.6])
            
            # Run a single optimization step to ensure the selection mode is used
            optimizer.optimize(1)

    def test_crossover(self, dummy_llm, base_optimizer_config, initial_prompts, dummy_task, dummy_predictor):
        """Test that the crossover method works correctly."""
        optimizer = EvoPromptGA(
            prompt_template="Combine these prompts to create a better one: <prompt1> and <prompt2>.",
            meta_llm=dummy_llm,
            selection_mode="random",
            config=base_optimizer_config,
            initial_prompts=initial_prompts,
            task=dummy_task,
            predictor=dummy_predictor
        )
        
        # Mock the task evaluation to make the tests deterministic  
        optimizer.scores = [0.8, 0.7, 0.6]
        
        # Call the crossover method
        child_prompts = optimizer._crossover(optimizer.prompts, optimizer.scores)
        
        # We should get the same number of child prompts as parents
        assert len(child_prompts) == len(initial_prompts)
        
        # Each child prompt should be a string
        for prompt in child_prompts:
            assert isinstance(prompt, str)

    def test_assertions(self, base_optimizer_config, initial_prompts, dummy_task):
        """Test that assertions are raised when required parameters are missing."""
        # Test assertion error when meta_llm is not provided
        with pytest.raises(AssertionError):
            optimizer = EvoPromptGA(
                prompt_template="Template",
                meta_llm=None,
                selection_mode="random",
                config=base_optimizer_config,
                initial_prompts=initial_prompts,
                task=dummy_task
            )
            
        # Test assertion error when invalid selection mode is provided
        with pytest.raises(AssertionError):
            optimizer = EvoPromptGA(
                prompt_template="Template",
                meta_llm=dummy_llm,
                selection_mode="invalid_mode",
                config=base_optimizer_config,
                initial_prompts=initial_prompts,
                task=dummy_task
            )