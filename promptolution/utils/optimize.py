"""The lightweight entry point: optimize prompts for a classification task in one call."""

from pathlib import Path

import pandas as pd

from typing import List, Optional, Union

from promptolution.utils.callbacks import FileOutputCallback
from promptolution.utils.evaluation import evaluate_prompts, train_test_split
from promptolution.utils.logging import get_logger
from promptolution.utils.runinfo import finish_runinfo, start_runinfo

logger = get_logger(__name__)

OPTIMIZERS = ("capo", "evopromptde", "evopromptga", "opro")


def _optimizer_class(name: str):
    """Resolve an optimizer name to its class.

    Imported lazily: pulling in the predictors/llms packages requires ``openai`` (the optional
    ``[api]`` extra), and merely importing promptolution must not.
    """
    from promptolution.optimizers import CAPO, OPRO, EvoPromptDE, EvoPromptGA

    return {"capo": CAPO, "evopromptde": EvoPromptDE, "evopromptga": EvoPromptGA, "opro": OPRO}[name]


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
    test_frac: float = 0.2,
    random_seed: int = 42,
    initial_prompts: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Optimize prompts for a classification task, with sensible defaults.

    Builds the components for you (an API LLM, a marker-based predictor, a classification task and
    the chosen optimizer), asking only for what the library cannot know: your data, an LLM and its
    credentials, and a description of the task. Runs in memory unless ``output_dir`` is given.

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
        test_frac (float): Fraction held out for the final evaluation; 0 evaluates on the train split.
        random_seed (int): Seed for the train/test split.
        initial_prompts (Optional[List[str]]): Prompts to start from. If omitted, they are generated
            from ``task_description``.
        output_dir (Optional[Union[str, Path]]): If set, also write the per-step trace, final scores
            and run info here. If omitted (the default), nothing is written to disk.

    Returns:
        pd.DataFrame: Columns ``prompt`` and ``score``, best first.
    """
    if task_type != "classification":
        raise NotImplementedError(
            f"optimize() supports classification only, got task_type={task_type!r}. For judge or "
            "reward tasks, build the components directly and call optimizer.optimize() "
            "(see the README quickstart)."
        )
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {optimizer!r}. Available: {', '.join(sorted(OPTIMIZERS))}.")

    # imported here, not at module level: these pull in openai (the optional [api] extra), and
    # merely importing promptolution must not require it
    from promptolution.llms.api_llm import APILLM
    from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor
    from promptolution.tasks.classification_tasks import ClassificationTask

    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    train_df, test_df = train_test_split(df, test_frac=test_frac, seed=random_seed)

    llm = APILLM(model_id=model_id, api_key=api_key, api_url=api_url)
    predictor = MarkerBasedPredictor(llm)
    # CAPO races prompts on data blocks, so the task needs a block eval_strategy
    train_task = ClassificationTask(
        train_df,
        task_description=task_description,
        x_column=x_column,
        y_column=y_column,
        eval_strategy="sequential_block",
    )
    test_task = (
        ClassificationTask(test_df, task_description=task_description, x_column=x_column, y_column=y_column)
        if test_frac > 0
        else train_task
    )

    opt = _optimizer_class(optimizer)(
        predictor=predictor, meta_llm=llm, task=train_task, initial_prompts=initial_prompts
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
        scores_df = evaluate_prompts(prompts, test_task, predictor)
    except Exception as e:
        if info is not None and output_dir is not None:
            finish_runinfo(output_dir, info, status="failed", error=str(e))
        raise

    if info is not None and output_dir is not None:
        scores_df.to_parquet(output_dir / "prompt_scores.parquet", index=False)
        finish_runinfo(output_dir, info, status="finished")
        logger.warning("✅ Finished run: %s", output_dir)
    return scores_df
