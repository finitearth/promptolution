"""Helper functions for the usage of the libary."""
from logging import Logger
from typing import List

import numpy as np
import pandas as pd

from promptolution.config import Config
from promptolution.exemplar_selectors import get_exemplar_selector
from promptolution.llms import get_llm
from promptolution.optimizers import get_optimizer
from promptolution.predictors import Classificator
from promptolution.tasks import get_task


def run_experiment(config: Config):
    """Run a full experiment based on the provided configuration.

    Args:
        config (Config): Configuration object for the experiment.

    Returns:
        pd.DataFrame: A DataFrame containing the prompts and their scores.
    """
    prompts = run_optimization(config)
    df = run_evaluation(config, prompts)
    return df


def run_optimization(config: Config):
    """Run the optimization phase of the experiment.

    Args:
        config (Config): Configuration object for the experiment.

    Returns:
        List[str]: The optimized list of prompts.
    """
    task = get_task(config)
    llm = get_llm(config.meta_llm, token=config.api_token)
    predictor = Classificator(llm, classes=task.classes)

    if config.init_pop_size:
        init_pop = np.random.choice(task.initial_population, size=config.init_pop_size, replace=True)
    else:
        init_pop = task.initial_population

    optimizer = get_optimizer(
        config,
        meta_llm=llm,
        initial_prompts=init_pop,
        task=task,
        predictor=predictor,
        n_eval_samples=config.n_eval_samples,
    )

    prompts = optimizer.optimize(n_steps=config.n_steps)

    if config.prepend_exemplars:
        selector = get_exemplar_selector(config.exemplar_selector, task, predictor)
        prompts = [selector.select_exemplars(p, n_examples=config.n_exemplars) for p in prompts]

    return prompts


def run_evaluation(config: Config, prompts: List[str]):
    """Run the evaluation phase of the experiment.

    Args:
        config (Config): Configuration object for the experiment.
        prompts (List[str]): List of prompts to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing the prompts and their scores.
    """
    task = get_task(config, split="test")

    llm = get_llm(config.evaluation_llm, token=config.api_token)
    predictor = Classificator(llm, classes=task.classes)

    scores = task.evaluate(prompts, predictor, subsample=True, n_samples=config.n_eval_samples)
    df = pd.DataFrame(dict(prompt=prompts, score=scores))
    df = df.sort_values("score", ascending=False)

    return df
