"""Promptolution: A framework for prompt optimization and a zoo of prompt optimization algorithms."""

# Import main configuration classes
from .config import ExperimentConfig
from .exemplar_selectors.base_exemplar_selector import BaseExemplarSelector
from .exemplar_selectors.random_search_selector import RandomSearchSelector
from .exemplar_selectors.random_selector import RandomSelector

# Re-export helper functions and factory methods
from .helpers import (
    get_exemplar_selector,
    get_llm,
    get_optimizer,
    get_predictor,
    get_task,
    run_evaluation,
    run_experiment,
    run_optimization,
)

# Re-export main classes for direct import
from .llms.api_llm import APILLM
from .llms.base_llm import BaseLLM
from .llms.local_llm import LocalLLM
from .llms.vllm import VLLM

# Import logging utilities
from .logging import get_logger, setup_logging
from .optimizers.base_optimizer import BaseOptimizer
from .optimizers.capo import CAPO
from .optimizers.evoprompt_de import EvoPromptDE
from .optimizers.evoprompt_ga import EvoPromptGA
from .optimizers.opro import Opro
from .predictors.base_predictor import BasePredictor
from .predictors.classifier import FirstOccurrenceClassifier, MarkerBasedClassifier
from .tasks.base_task import BaseTask
from .tasks.classification_tasks import ClassificationTask

# Re-export templates for direct access
from .templates import (
    CAPO_CROSSOVER_TEMPLATE,
    CAPO_MUTATION_TEMPLATE,
    EVOPROMPT_DE_TEMPLATE,
    EVOPROMPT_DE_TEMPLATE_TD,
    EVOPROMPT_GA_TEMPLATE,
    EVOPROMPT_GA_TEMPLATE_TD,
    OPRO_TEMPLATE,
    OPRO_TEMPLATE_TD,
)
