"""Module for prompt optimizers."""

from promptolution.templates import (
    EVOPROMPT_DE_TEMPLATE,
    EVOPROMPT_DE_TEMPLATE_TD,
    EVOPROMPT_GA_TEMPLATE,
    EVOPROMPT_GA_TEMPLATE_TD,
    OPRO_TEMPLATE,
)

from .base_optimizer import DummyOptimizer
from .evoprompt_de import EvoPromptDE
from .evoprompt_ga import EvoPromptGA
from .opro import Opro


def get_optimizer(
    config=None, optimizer: str = None, include_task_desc: bool = None, meta_prompt: str = None, *args, **kwargs
):
    """Factory function to create and return an optimizer instance based on the provided configuration.

    This function selects and instantiates the appropriate optimizer class based on the
    'optimizer' field in the config object. Alternatively you can pass the relevant parameters.
    It supports three types of optimizers: 'dummy', 'evopromptde', 'evopromptga', and 'opro'.

    Args:
        config (Config): Configuration object containing the optimizer type.
        optimizer (str): Identifier for the optimizer to use. Special cases:
                         - "dummy" for DummyOptimizer
                         - Any other string for the specified optimizer class
        include_task_desc (bool): Flag to include task description in the prompt.
        meta_prompt (str): Meta prompt for the optimizer.
        *args: Variable length argument list passed to the optimizer constructor.
        **kwargs: Arbitrary keyword arguments passed to the optimizer constructor

    Returns:
        An instance of the specified optimizer class.

    Raises:
        ValueError: If an unknown optimizer type is specified in the config.
    """
    if config.optimizer == "dummy":
        return DummyOptimizer(*args, **kwargs)

    if optimizer is None:
        optimizer = config.optimizer

    if include_task_desc is None:
        include_task_desc = config.include_task_desc

    if config is not None and meta_prompt is None:
        meta_prompt = config.meta_prompt

    if config.optimizer == "evopromptde":
        prompt_template = EVOPROMPT_DE_TEMPLATE_TD if include_task_desc else EVOPROMPT_DE_TEMPLATE
        prompt_template = meta_prompt if meta_prompt else prompt_template
        donor_random = kwargs.get("donor_random", config.donor_random if config is not None else None)
        return EvoPromptDE(donor_random=donor_random, prompt_template=prompt_template, *args, **kwargs)

    if config.optimizer == "evopromptga":
        prompt_template = EVOPROMPT_GA_TEMPLATE_TD if config.include_task_desc else EVOPROMPT_GA_TEMPLATE
        prompt_template = config.meta_prompt if meta_prompt else prompt_template
        selection_mode = kwargs.get("selection_mode", config.selection_mode if config is not None else None)
        return EvoPromptGA(selection_mode=selection_mode, prompt_template=prompt_template, *args, **kwargs)

    if config.optimizer == "opro":
        prompt_template = OPRO_TEMPLATE
        prompt_template = config.meta_prompt if config.meta_prompt else prompt_template
        n_samples = kwargs.get("n_samples", config.n_samples if config is not None else None)
        return Opro(prompt_template=prompt_template, n_samples=n_samples, *args, **kwargs)

    raise ValueError(f"Unknown optimizer: {config.optimizer}")
