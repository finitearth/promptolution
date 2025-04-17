"""Helper functions for the usage of the libary."""

from logging import getLogger
from typing import List

import pandas as pd

from promptolution.callbacks import LoggerCallback
from promptolution.config import ExperimentConfig
from promptolution.exemplar_selectors import get_exemplar_selector
from promptolution.llms import get_llm
from promptolution.optimizers import get_optimizer
from promptolution.predictors import get_predictor
from promptolution.tasks import get_task

logger = getLogger(__name__)


def run_experiment(df: pd.DataFrame, config: ExperimentConfig):
    """Run a full experiment based on the provided configuration.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        config (Config): Configuration object for the experiment.

    Returns:
        pd.DataFrame: A DataFrame containing the prompts and their scores.
    """
    # train test split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    prompts = run_optimization(train_df, config)
    df_prompt_scores = run_evaluation(test_df, config, prompts)

    return df_prompt_scores


def run_optimization(df, config: ExperimentConfig):
    """Run the optimization phase of the experiment.

    Args:
        config (Config): Configuration object for the experiment.

    Returns:
        List[str]: The optimized list of prompts.
    """
    llm = get_llm(config=config)
    predictor = get_predictor(llm, config=config)
    config.task_description = config.task_description + " " + predictor.extraction_description

    task = get_task(df, config)
    optimizer = get_optimizer(
        predictor=predictor,
        meta_llm=llm,
        task=task,
        config=config,
    )

    prompts = optimizer.optimize(n_steps=config.n_steps)

    if config.prepend_exemplars:
        selector = get_exemplar_selector(config.exemplar_selector, task, predictor)
        prompts = [selector.select_exemplars(p, n_examples=config.n_exemplars) for p in prompts]

    return prompts


def run_evaluation(df: pd.DataFrame, config: ExperimentConfig, prompts: List[str]):
    """Run the evaluation phase of the experiment.

    Args:
        config (Config): Configuration object for the experiment.
        prompts (List[str]): List of prompts to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing the prompts and their scores.
    """
    task = get_task(df, config)

    llm = get_llm(config)
    predictor = get_predictor(llm, classes=task.classes)

    scores = task.evaluate(prompts, predictor)
    df = pd.DataFrame(dict(prompt=prompts, score=scores))
    df = df.sort_values("score", ascending=False)

    return df
