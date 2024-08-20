from .base_optimizer import DummyOptimizer
from .evoprompt_de import EvoPromptDE
from .evoprompt_ga import EvoPromptGA


def get_optimizer(config, *args, **kwargs):
    if config.optimizer == "dummy":
        return DummyOptimizer(*args, **kwargs)
    if config.optimizer == "evopromptde":
        return EvoPromptDE(donor_random=config.donor_random, *args, **kwargs)
    if config.optimizer == "evopromptga":
        return EvoPromptGA(selection_mode=config.selection_mode, *args, **kwargs)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")
