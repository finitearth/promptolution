"""Promptolution: A library for prompt tuning.

This library provides tools and utilities for optimizing prompts for language models.
It follows the Hugging Face-style interfaces for configuration validation.
"""

from . import callbacks, config, llms, optimizers, predictors, tasks

# Import main configuration classes
from .config import BaseConfig, PromptolutionConfig

# Import factory functions
from .llms import get_llm
from .llms.base_llm import BaseLLM, LLMModelConfig
from .optimizers import get_optimizer
from .optimizers.base_optimizer import BaseOptimizer, OptimizerConfig
from .predictors import get_predictor
from .predictors.base_predictor import BasePredictor, PredictorConfig
from .tasks import get_task
from .tasks.base_task import BaseTask, TaskConfig

__version__ = "1.3.0"
