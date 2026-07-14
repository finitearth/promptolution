"""Promptolution: a framework for prompt optimization and a zoo of prompt optimization algorithms.

The end-user API is two functions: :func:`optimize` (find the best prompt for your task) and
:func:`evaluate` (score existing prompts on your data). Components (`llms`, `tasks`, `predictors`,
`optimizers`) are importable for full manual control; config-driven runs and whole grids live in
`promptolution.experiment_grid`.
"""

from promptolution.api import evaluate, optimize

__all__ = ["optimize", "evaluate"]
