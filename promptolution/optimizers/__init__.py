"""Module for prompt optimizers."""

from typing import Literal

from promptolution.config import ExperimentConfig
from promptolution.llms.base_llm import BaseLLM
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask
from promptolution.templates import (
    CAPO_CROSSOVER_TEMPLATE,
    CAPO_MUTATION_TEMPLATE,
    EVOPROMPT_DE_TEMPLATE,
    EVOPROMPT_DE_TEMPLATE_TD,
    EVOPROMPT_GA_TEMPLATE,
    EVOPROMPT_GA_TEMPLATE_TD,
    OPRO_TEMPLATE,
    OPRO_TEMPLATE_TD,
)

from .capo import CAPO
from .evoprompt_de import EvoPromptDE
from .evoprompt_ga import EvoPromptGA
from .opro import Opro
