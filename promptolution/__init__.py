"""Promptolution: A library for prompt tuning.

This library provides tools and utilities for optimizing prompts for language models.
It follows the Hugging Face-style interfaces for configuration validation.
"""

from . import callbacks, config, llms, optimizers, predictors, tasks

# Import main configuration classes
from .config import ExperimentConfig
from .helpers import run_evaluation, run_experiment, run_optimization

# Import factory functions
from .llms import get_llm
from .llms.api_llm import APILLM
from .llms.base_llm import BaseLLM
from .llms.local_llm import LocalLLM
from .llms.vllm import VLLM
from .optimizers import get_optimizer
from .optimizers.base_optimizer import BaseOptimizer
from .predictors import get_predictor
from .predictors.base_predictor import BasePredictor
from .tasks import get_task
from .tasks.base_task import BaseTask

__version__ = "1.3.0"
