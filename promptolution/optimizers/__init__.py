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


def get_optimizer(config, *args, **kwargs):
    """Factory function to create and return an optimizer instance based on the provided configuration.

    This function selects and instantiates the appropriate optimizer class based on the
    'optimizer' field in the config object. It supports three types of optimizers:
    'dummy', 'evopromptde', and 'evopromptga'.

    Args:
        config: A configuration object that must have an 'optimizer' attribute.
                For 'evopromptde', it should also have a 'donor_random' attribute.
                For 'evopromptga', it should also have a 'selection_mode' attribute.
        *args: Variable length argument list passed to the optimizer constructor.
        **kwargs: Arbitrary keyword arguments passed to the optimizer constructor.

    Returns:
        An instance of the specified optimizer class.

    Raises:
        ValueError: If an unknown optimizer type is specified in the config.
    """
    if config.optimizer == "dummy":
        return DummyOptimizer(*args, **kwargs)
    if config.optimizer == "evopromptde":
        prompt_template = EVOPROMPT_DE_TEMPLATE_TD if config.include_task_desc else EVOPROMPT_DE_TEMPLATE
        prompt_template = config.meta_prompt if config.meta_prompt else prompt_template
        return EvoPromptDE(donor_random=config.donor_random, prompt_template=prompt_template, *args, **kwargs)
    if config.optimizer == "evopromptga":
        prompt_template = EVOPROMPT_GA_TEMPLATE_TD if config.include_task_desc else EVOPROMPT_GA_TEMPLATE
        prompt_template = config.meta_prompt if config.meta_prompt else prompt_template
        return EvoPromptGA(selection_mode=config.selection_mode, prompt_template=prompt_template, *args, **kwargs)
    if config.optimizer == "opro":
        prompt_template = OPRO_TEMPLATE
        prompt_template = config.meta_prompt if config.meta_prompt else prompt_template
        return Opro(prompt_template=prompt_template, *args, **kwargs)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")
