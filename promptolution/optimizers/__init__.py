"""Module for prompt optimizers."""

from promptolution.optimizers.capo import CAPO
from promptolution.optimizers.evoprompt_de import EvoPromptDE
from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.optimizers.opro import OPRO
from promptolution.optimizers.templates import (
    CAPO_CROSSOVER_TEMPLATE,
    CAPO_DOWNSTREAM_TEMPLATE,
    CAPO_FEWSHOT_TEMPLATE,
    CAPO_MUTATION_TEMPLATE,
    DEFAULT_SYS_PROMPT,
    EVOPROMPT_DE_TEMPLATE,
    EVOPROMPT_DE_TEMPLATE_TD,
    EVOPROMPT_GA_TEMPLATE,
    EVOPROMPT_GA_TEMPLATE_TD,
    OPRO_TEMPLATE,
    OPRO_TEMPLATE_TD,
    PROMPT_CREATION_TEMPLATE,
    PROMPT_CREATION_TEMPLATE_TD,
    PROMPT_VARIATION_TEMPLATE,
)
