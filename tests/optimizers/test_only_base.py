"""Minimal test for the base optimizer."""

import pytest
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from promptolution.optimizers.base_optimizer import OptimizerConfig

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