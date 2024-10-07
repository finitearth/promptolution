"""Helper functions for the usage of the libary."""
from typing import List

import pandas as pd

from promptolution.config import Config
from promptolution.llms import get_llm
from promptolution.optimizers import get_optimizer
from promptolution.predictors.classificator import Classificator
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

    optimizer = get_optimizer(
        config, meta_llm=llm, initial_prompts=task.initial_population, task=task, predictor=predictor
    )

    prompts = optimizer.optimize(n_steps=config.n_steps)

    return prompts


def run_evaluation(config: Config, prompts: List[str]):
    """Run the evaluation phase of the experiment.

    Args:
        config (Config): Configuration object for the experiment.
        prompts (List[str]): List of prompts to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing the prompts and their scores.
    """
    task = get_task(config)

    token = open("../deepinfratoken.txt", "r").read()
    llm = get_llm(config.evaluation_llm, token=token)

    predictor = Classificator(llm, classes=task.classes)

    scores = task.evaluate(prompts, predictor)

    df = pd.DataFrame(dict(prompt=prompts, score=scores))

    return df
