"""The end-user API: optimize prompts for a classification task, evaluate prompts on your data.

Both functions require only what the library cannot know as arguments: the LLM (with its credentials), the data,
and (for optimization) what the task is. Everything else has working defaults, overridable per call.
"""

import pandas as pd

from typing import TYPE_CHECKING, Callable, List, Optional, Union

from promptolution.optimizers.capo import CAPO
from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.evaluation import score_prompts, train_test_split
from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.optimizers.base_optimizer import BaseOptimizer

logger = get_logger(__name__)


def optimize(
    llm: "BaseLLM",
    df: pd.DataFrame,
    task_description: str,
    *,
    x_column: str = "x",
    y_column: str = "y",
    optimizer: Callable[..., "BaseOptimizer"] = CAPO,
    initial_prompts: Optional[List[str]] = None,
    n_steps: int = 10,
    test_frac: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Find the best prompt for a classification task.

    The data is split, prompts are optimized with CAPO (cost-aware, evaluating on data blocks)
    starting from prompts generated out of the task description, and the result is scored exactly
    on the held-out split.

    Args:
        llm (BaseLLM): The language model, used both to run prompts and to propose new ones.
        df (pd.DataFrame): The data (or a df-like with ``.to_pandas()``).
        task_description (str): What the task is, including the possible labels.
        x_column (str): Column with the input texts.
        y_column (str): Column with the labels.
        optimizer (Callable[..., BaseOptimizer]): Optimizer class or factory, called with
            ``predictor``, ``meta_llm``, ``task`` and ``initial_prompts``; e.g. ``EvoPromptGA`` or
            ``functools.partial(CAPO, upper_shots=3)``.
        initial_prompts (Optional[List[str]]): Starting prompts; generated from ``task_description``
            if not given.
        n_steps (int): Number of optimization steps.
        test_frac (float): Held-out fraction for the final evaluation; 0 evaluates on the train split.
        seed (int): Random seed for splitting and subsampling.

    Returns:
        pd.DataFrame: Columns ``prompt`` and ``score``, best first.
    """
    train_df, test_df = train_test_split(df, test_frac=test_frac, seed=seed)
    train_task = ClassificationTask(
        train_df,
        task_description=task_description,
        x_column=x_column,
        y_column=y_column,
        eval_strategy="sequential_block",
        seed=seed,
    )
    predictor = MarkerBasedPredictor(llm)
    opt = optimizer(predictor=predictor, meta_llm=llm, task=train_task, initial_prompts=initial_prompts)

    logger.warning("🔥 Starting optimization...")
    prompts = opt.optimize(n_steps=n_steps)

    if test_frac > 0:
        test_task = ClassificationTask(test_df, task_description=task_description, x_column=x_column, y_column=y_column)
    else:
        test_task = train_task
    return score_prompts(prompts, test_task, predictor)


def evaluate(
    llm: "BaseLLM",
    df: pd.DataFrame,
    prompts: Union[List[Prompt], List[str]],
    *,
    x_column: str = "x",
    y_column: str = "y",
) -> pd.DataFrame:
    """Score existing prompts on your data — e.g. prompts saved from an earlier :func:`optimize` run.

    Args:
        llm (BaseLLM): The language model that runs the prompts.
        df (pd.DataFrame): The data to score on (all rows are used; split beforehand if you need to).
        prompts (Union[List[Prompt], List[str]]): The prompts to score.
        x_column (str): Column with the input texts.
        y_column (str): Column with the labels.

    Returns:
        pd.DataFrame: Columns ``prompt`` and ``score``, best first.
    """
    task = ClassificationTask(df, x_column=x_column, y_column=y_column)
    return score_prompts(prompts, task, MarkerBasedPredictor(llm))
