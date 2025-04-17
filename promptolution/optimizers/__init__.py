"""Module for prompt optimizers."""

from promptolution.config import ExperimentConfig
from promptolution.templates import (
    EVOPROMPT_DE_TEMPLATE,
    EVOPROMPT_DE_TEMPLATE_TD,
    EVOPROMPT_GA_TEMPLATE,
    EVOPROMPT_GA_TEMPLATE_TD,
    OPRO_TEMPLATE,
    OPRO_TEMPLATE_TD,
)

from .base_optimizer import DummyOptimizer
from .evoprompt_de import EvoPromptDE
from .evoprompt_ga import EvoPromptGA
from .opro import Opro


def get_optimizer(
    predictor, meta_llm, task, optimizer=None, meta_prompt=None, task_description=None, config: ExperimentConfig = None
):
    """Creates and returns an optimizer instance based on provided parameters.

    Args:
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

    if optimizer == "dummy":
        return DummyOptimizer(predictor=predictor, config=config)

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

    raise ValueError(f"Unknown optimizer: {config.optimizer}")
