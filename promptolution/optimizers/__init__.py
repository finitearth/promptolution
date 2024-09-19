from .base_optimizer import DummyOptimizer
from .evoprompt_de import EvoPromptDE
from .evoprompt_ga import EvoPromptGA


def get_optimizer(config, *args, **kwargs):
    """
    Factory function to create and return an optimizer instance based on the provided configuration.

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
        return EvoPromptDE(donor_random=config.donor_random, *args, **kwargs)
    if config.optimizer == "evopromptga":
        return EvoPromptGA(selection_mode=config.selection_mode, *args, **kwargs)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")
