"""Tests for the BaseOptimizer class."""

import pytest
import numpy as np
from typing import List

from promptolution.optimizers.base_optimizer import BaseOptimizer, OptimizerConfig, DummyOptimizer
from promptolution.config import PromptolutionConfig


class MockCallback:
    """Mock callback for testing callback functionality."""
    
    def __init__(self):
        """Initialize the mock callback."""
        self.on_step_end_called = 0
        self.on_epoch_end_called = 0
        self.on_train_end_called = 0
        self.return_value = True
    
    def on_step_end(self, optimizer):
        """Record that on_step_end was called."""
        self.on_step_end_called += 1
        return self.return_value
    
    def on_epoch_end(self, optimizer):
        """Record that on_epoch_end was called."""
        self.on_epoch_end_called += 1
        return self.return_value
    
    def on_train_end(self, optimizer):
        """Record that on_train_end was called."""
        self.on_train_end_called += 1
        return True


class TestBaseOptimizer:
    """Test suite for the BaseOptimizer class."""

    def test_init_with_dict_config(self, initial_prompts, dummy_task):
        """Test initialization with a dictionary config."""
        config_dict = {
            "optimizer_name": "test_optimizer",
            "n_steps": 10,
            "population_size": 12,
            "random_seed": 42
        }
        
        # Create a concrete optimizer since BaseOptimizer is abstract
        optimizer = DummyOptimizer(
            initial_prompts=initial_prompts,
            task=dummy_task,
            config=config_dict
        )
        
        assert optimizer.prompts == initial_prompts
        assert optimizer.task == dummy_task

    def test_init_with_object_config(self, base_optimizer_config, initial_prompts, dummy_task):
        """Test initialization with an OptimizerConfig object."""
        optimizer = DummyOptimizer(
            initial_prompts=initial_prompts,
            task=dummy_task,
            config=base_optimizer_config
        )
        
        assert optimizer.prompts == initial_prompts
        assert optimizer.task == dummy_task

    def test_callbacks(self, initial_prompts):
        """Test that callbacks are properly called."""
        callback = MockCallback()
        
        optimizer = DummyOptimizer(
            initial_prompts=initial_prompts,
            callbacks=[callback]
        )
        
        # Optimize should call all the callback methods
        optimizer.optimize(n_steps=5)
        
        assert callback.on_step_end_called == 1
        assert callback.on_epoch_end_called == 1
        assert callback.on_train_end_called == 1

    def test_early_stopping(self, initial_prompts):
        """Test that optimization stops early if a callback returns False."""
        callback = MockCallback()
        callback.return_value = False
        
        optimizer = DummyOptimizer(
            initial_prompts=initial_prompts,
            callbacks=[callback]
        )
        
        # The optimization should stop early but still call the on_train_end method
        optimizer.optimize(n_steps=5)
        
        assert callback.on_step_end_called == 1
        assert callback.on_epoch_end_called == 0  # Won't be called because we stopped early
        assert callback.on_train_end_called == 1

    def test_dummy_optimizer(self, initial_prompts):
        """Test that DummyOptimizer works correctly."""
        optimizer = DummyOptimizer(initial_prompts)
        
        # DummyOptimizer should return the initial prompts unchanged
        result = optimizer.optimize(n_steps=5)
        
        assert result == initial_prompts