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

from .base_optimizer import DummyOptimizer
from .capo import CAPO
from .evoprompt_de import EvoPromptDE
from .evoprompt_ga import EvoPromptGA
from .opro import Opro


def get_optimizer(
    predictor: BasePredictor,
    meta_llm: BaseLLM,
    task: BaseTask,
    optimizer: Literal["evopromptde", "evopromptga", "opro"] = None,
    meta_prompt: str = None,
    task_description: str = None,
    config: ExperimentConfig = None,
):
    """Creates and returns an optimizer instance based on provided parameters.

    Args:
        predictor: The predictor used for prompt evaluation
        meta_llm: The language model used for generating meta-prompts
        task: The task object used for evaluating prompts
        optimizer: String identifying which optimizer to use
        meta_prompt: Meta prompt text for the optimizer
        task_description: Description of the task for the optimizer
        config: Configuration object with default parameters

    Returns:
        An optimizer instance

    Raises:
        ValueError: If an unknown optimizer type is specified
    """
    if optimizer is None:
        optimizer = config.optimizer
    if task_description is None:
        task_description = config.task_description
    if meta_prompt is None and hasattr(config, "meta_prompt"):
        meta_prompt = config.meta_prompt

    if config.optimizer == "capo":
        crossover_template = (
            CAPO_CROSSOVER_TEMPLATE.replace("<task_desc>", task_description)
            if task_description
            else CAPO_CROSSOVER_TEMPLATE
        )
        mutation_template = (
            CAPO_MUTATION_TEMPLATE.replace("<task_desc>", task_description)
            if task_description
            else CAPO_MUTATION_TEMPLATE
        )

        return CAPO(
            predictor=predictor,
            meta_llm=meta_llm,
            task=task,
            crossover_template=crossover_template,
            mutation_template=mutation_template,
            config=config,
        )

    if config.optimizer == "evopromptde":
        template = (
            EVOPROMPT_DE_TEMPLATE_TD.replace("<task_desc>", task_description)
            if task_description
            else EVOPROMPT_DE_TEMPLATE
        )
        return EvoPromptDE(predictor=predictor, meta_llm=meta_llm, task=task, prompt_template=template, config=config)

    if config.optimizer == "evopromptga":
        template = (
            EVOPROMPT_GA_TEMPLATE_TD.replace("<task_desc>", task_description)
            if task_description
            else EVOPROMPT_GA_TEMPLATE
        )
        return EvoPromptGA(predictor=predictor, meta_llm=meta_llm, task=task, prompt_template=template, config=config)

    if config.optimizer == "opro":
        template = OPRO_TEMPLATE_TD.replace("<task_desc>", task_description) if task_description else OPRO_TEMPLATE
        return Opro(predictor=predictor, meta_llm=meta_llm, task=task, prompt_template=template, config=config)

    if optimizer == "dummy":
        return DummyOptimizer(predictor=predictor, config=config)

    raise ValueError(f"Unknown optimizer: {config.optimizer}")
