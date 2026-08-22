"""The lightweight entry points: optimize prompts on a dataset, and score them on another."""

from pathlib import Path

import pandas as pd

from typing import Dict, List, Literal, Optional, Type, Union

from promptolution.llms.api_llm import APILLM
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.optimizers.capo import CAPO
from promptolution.optimizers.evoprompt_de import EvoPromptDE
from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.optimizers.opro import OPRO
from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.callbacks import FileOutputCallback
from promptolution.utils.evaluation import evaluate_prompts
from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt
from promptolution.utils.runinfo import finish_runinfo, start_runinfo

logger = get_logger(__name__)

OPTIMIZERS: Dict[str, Type[BaseOptimizer]] = {
    "capo": CAPO,
    "evopromptde": EvoPromptDE,
    "evopromptga": EvoPromptGA,
    "opro": OPRO,
}


def _check_task_type(task_type: str, caller: str) -> None:
    if task_type != "classification":
        raise NotImplementedError(
            f"{caller}() supports classification only, got task_type={task_type!r}. For judge or "
            "reward tasks, build the components directly (see the README quickstart)."
        )


def _build(
    data: Union[pd.DataFrame, str, Path],
    task_description: str,
    model_id: str,
    api_key: Optional[str],
    api_url: Optional[str],
    x_column: str,
    y_column: str,
    eval_strategy: Literal["full", "subsample", "sequential_block", "random_block"],
):
    """Build the LLM, predictor and classification task shared by optimize() and evaluate()."""
    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    llm = APILLM(model_id=model_id, api_key=api_key, api_url=api_url)
    predictor = MarkerBasedPredictor(llm)
    task = ClassificationTask(
        df,
        task_description=task_description,
        x_column=x_column,
        y_column=y_column,
        eval_strategy=eval_strategy,
    )
    return llm, predictor, task


def optimize(
    data: Union[pd.DataFrame, str, Path],
    task_description: str,
    task_type: str = "classification",
    *,
    model_id: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    x_column: str = "x",
    y_column: str = "y",
    optimizer: str = "capo",
    n_steps: int = 10,
    initial_prompts: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> List[Prompt]:
    """Optimize prompts on a dataset, with sensible defaults.

    Builds the components for you (an API LLM, a marker-based predictor, a classification task and
    the chosen optimizer), asking only for what the library cannot know: your data, an LLM and its
    credentials, and a description of the task.

    All of ``data`` is used to select prompts, so pass your dev split here and keep a held-out split
    for :func:`evaluate`. Nothing is fitted on this data in the gradient sense; the optimizer only
    uses it to score and select candidate prompts.

    Only classification is supported. For judge or reward tasks, or for full control over the
    components, build them yourself and call ``optimizer.optimize()`` (see the README). For
    config-driven runs, grids, SLURM and restart, use the ``promptolution-experiment`` CLI.

    Args:
        data (Union[pd.DataFrame, str, Path]): A DataFrame, or anything ``pandas.read_csv`` accepts.
        task_description (str): Description of the task, used to generate initial prompts and passed
            to the meta-LLM during optimization.
        task_type (str): Only ``"classification"`` is supported; anything else raises.
        model_id (str): Model id of the LLM used for both predictions and prompt proposals.
        api_key (Optional[str]): API key. If omitted, the underlying OpenAI client resolves it from
            the environment (e.g. ``OPENAI_API_KEY``).
        api_url (Optional[str]): API base URL, for providers other than OpenAI.
        x_column (str): Input column name in ``data``.
        y_column (str): Label column name in ``data``.
        optimizer (str): One of ``capo``, ``evopromptde``, ``evopromptga``, ``opro``.
        n_steps (int): Number of optimization steps.
        initial_prompts (Optional[List[str]]): Prompts to start from. If omitted, they are generated
            from ``task_description``.
        output_dir (Optional[Union[str, Path]]): If set, write the per-step trace and run info here.
            If omitted (the default), nothing is written to disk.

    Returns:
        List[Prompt]: The optimized prompts. Pass them to :func:`evaluate` to score them.
    """
    _check_task_type(task_type, "optimize")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {optimizer!r}. Available: {', '.join(sorted(OPTIMIZERS))}.")

    # CAPO races prompts on data blocks, so the task needs a block eval_strategy
    llm, predictor, task = _build(
        data, task_description, model_id, api_key, api_url, x_column, y_column, "sequential_block"
    )

    # meta_llm is accepted by every concrete optimizer but not declared on BaseOptimizer, hence the ignore
    opt = OPTIMIZERS[optimizer](
        predictor=predictor, meta_llm=llm, task=task, initial_prompts=initial_prompts  # type: ignore[call-arg]
    )

    info = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        opt.callbacks.append(FileOutputCallback(dir=output_dir))
        info = start_runinfo(output_dir, output_dir.name)

    try:
        logger.warning("🔥 Starting optimization...")
        prompts = opt.optimize(n_steps=n_steps)
    except Exception as e:
        if info is not None and output_dir is not None:
            finish_runinfo(output_dir, info, status="failed", error=str(e))
        raise

    if info is not None and output_dir is not None:
        finish_runinfo(output_dir, info, status="finished")
        logger.warning("✅ Finished optimization: %s", output_dir)
    return prompts


def evaluate(
    prompts: Union[List[Prompt], List[str]],
    data: Union[pd.DataFrame, str, Path],
    task_description: str,
    task_type: str = "classification",
    *,
    model_id: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    x_column: str = "x",
    y_column: str = "y",
) -> pd.DataFrame:
    """Score prompts on a dataset and return a sorted prompt/score table.

    The counterpart to :func:`optimize`: pass the prompts it returned and your held-out split.
    Evaluation runs on all of ``data``.

    Args:
        prompts (Union[List[Prompt], List[str]]): Prompts to score, e.g. the output of :func:`optimize`.
        data (Union[pd.DataFrame, str, Path]): A DataFrame, or anything ``pandas.read_csv`` accepts.
        task_description (str): Description of the task.
        task_type (str): Only ``"classification"`` is supported; anything else raises.
        model_id (str): Model id of the LLM used for predictions.
        api_key (Optional[str]): API key. If omitted, resolved from the environment.
        api_url (Optional[str]): API base URL, for providers other than OpenAI.
        x_column (str): Input column name in ``data``.
        y_column (str): Label column name in ``data``.

    Returns:
        pd.DataFrame: Columns ``prompt`` and ``score``, best first.
    """
    _check_task_type(task_type, "evaluate")
    _, predictor, task = _build(data, task_description, model_id, api_key, api_url, x_column, y_column, "full")
    return evaluate_prompts(prompts, task, predictor)
