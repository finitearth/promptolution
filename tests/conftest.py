"""Test fixtures for promptolution tests."""

import pytest
import numpy as np
from typing import List

from promptolution.optimizers.base_optimizer import BaseOptimizer, OptimizerConfig
from promptolution.tasks.base_task import BaseTask, DummyTask
from promptolution.llms.base_llm import BaseLLM
from promptolution.predictors.base_predictor import DummyPredictor


class DummyLLM(BaseLLM):
    """A dummy LLM for testing purposes."""

    def __init__(self, *args, **kwargs):
        """Initialize the DummyLLM."""
        self.model_id = "dummy"
        self.responses = [
            "<prompt>This is an optimized prompt.</prompt>",
            "<prompt>This is another optimized prompt.</prompt>",
            "<prompt>This is a third optimized prompt.</prompt>",
        ]
        
    def get_response(self, prompts: List[str]) -> List[str]:
        """Return dummy responses."""
        return self.responses[:len(prompts)]


@pytest.fixture
def dummy_task():
    """Return a dummy task for testing."""
    return DummyTask()


@pytest.fixture
def dummy_predictor():
    """Return a dummy predictor for testing."""
    return DummyPredictor(classes=["positive", "negative"])


@pytest.fixture
def dummy_llm():
    """Return a dummy LLM for testing."""
    return DummyLLM()


@pytest.fixture
def base_optimizer_config():
    """Return a basic optimizer configuration."""
    return OptimizerConfig(
        optimizer_name="test_optimizer",
        n_steps=5,
        population_size=8,
        random_seed=42,
        n_eval_samples=10
    )


@pytest.fixture
def initial_prompts():
    """Return a list of initial prompts for testing."""
    return [
        "Classify the following text as positive or negative.",
        "Determine if the sentiment of the text is positive or negative.",
        "Is the following text positive or negative?",
    ]