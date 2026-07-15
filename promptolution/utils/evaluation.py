"""Component-level evaluation helpers: split off a held-out set, score prompts on a task."""

import pandas as pd

from typing import TYPE_CHECKING, List, Tuple, Union, cast

from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.tasks.base_task import BaseTask

logger = get_logger(__name__)


def train_test_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into a train and test set.

    Args:
        df (pd.DataFrame): The data to split (or a df-like with ``.to_pandas()``).
        test_frac (float): Fraction of rows held out for the test split.
        seed (int): Random seed for the split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: ``(train_df, test_df)``, both with a fresh index.
    """
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    test_df = df.sample(frac=test_frac, random_state=seed)
    train_df = df.drop(test_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def score_prompts(
    prompts: Union[List[Prompt], List[str]], task: "BaseTask", predictor: "BasePredictor"
) -> pd.DataFrame:
    """Score prompts on a task and return a sorted prompt/score table.

    Args:
        prompts (Union[List[Prompt], List[str]]): Prompts to score.
        task (BaseTask): The task to score on (its full dataset is used).
        predictor (BasePredictor): The predictor that runs the prompts.

    Returns:
        pd.DataFrame: Columns ``prompt`` and ``score``, best first.
    """
    if isinstance(prompts[0], str):
        str_prompts = cast(List[str], list(prompts))
        prompt_objs = [Prompt(p) for p in str_prompts]
    else:
        prompt_objs = cast(List[Prompt], list(prompts))
        str_prompts = [p.construct_prompt() for p in prompt_objs]
    logger.warning("📊 Starting evaluation...")
    results = task.evaluate(prompt_objs, predictor, eval_strategy="full")
    return pd.DataFrame({"prompt": str_prompts, "score": results.agg_scores.tolist()}).sort_values(
        "score", ascending=False, ignore_index=True
    )
