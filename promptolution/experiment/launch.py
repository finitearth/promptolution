"""Build and run experiment cells: Hydra CLI entry point, and the lightweight `optimize()`."""

from pathlib import Path

import hydra
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from typing import Any, Dict, List, Optional, Union

from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.callbacks import FileOutputCallback
from promptolution.utils.evaluation import evaluate_prompts, train_test_split
from promptolution.utils.logging import get_logger
from promptolution.utils.runinfo import finish_runinfo, start_runinfo

logger = get_logger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parent / "conf")
FINISHED_MARKER = ".finished"


def build_components(cfg, train_task, out_dir: Optional[Union[str, Path]] = None):
    """Instantiate llm, (meta_llm,) predictor and optimizer from cfg for a given train task.

    Args:
        cfg: A composed experiment config (see ``conf/config.yaml`` for the schema).
        train_task: The task the optimizer runs against.
        out_dir (Optional[Union[str, Path]]): If set, attach a FileOutputCallback writing
            per-step results here.

    Returns:
        A ``(predictor, optimizer)`` pair, ready for ``optimizer.optimize(n_steps=...)``.
    """
    llm = instantiate(cfg.llm)
    meta_llm = instantiate(cfg.meta_llm) if cfg.meta_llm else llm  # null in config -> share the llm
    predictor = instantiate(cfg.predictor, llm=llm)
    optimizer = instantiate(cfg.optimizer, predictor=predictor, meta_llm=meta_llm, task=train_task)
    if out_dir is not None:
        optimizer.callbacks.append(FileOutputCallback(dir=out_dir))
    return predictor, optimizer


def _compose(overrides: Optional[List[str]] = None, config_name: str = "config"):
    """Compose the shipped conf/ without the CLI (Hydra only auto-composes under @hydra.main)."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides or [])


def _run(cfg, train_task, test_task, out_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Build components, optimize, evaluate; write the output contract only if out_dir is set."""
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.skip_completed and (out_dir / FINISHED_MARKER).exists():
            logger.warning("⏭️  Skipping finished run: %s", out_dir)
            return pd.read_parquet(out_dir / "prompt_scores.parquet")

    predictor, optimizer = build_components(cfg, train_task, out_dir=out_dir)
    info: Optional[Dict[str, Any]] = None
    if out_dir is not None:
        info = start_runinfo(out_dir, cfg.name)

    try:
        logger.warning("🔥 Starting optimization...")
        prompts = optimizer.optimize(n_steps=cfg.n_steps)
        scores_df = evaluate_prompts(prompts, test_task, predictor)
    except Exception as e:
        if out_dir is not None and info is not None:
            finish_runinfo(out_dir, info, status="failed", error=str(e))
        raise

    if out_dir is not None and info is not None:
        scores_df.to_parquet(out_dir / "prompt_scores.parquet", index=False)
        finish_runinfo(out_dir, info, status="finished")
        (out_dir / FINISHED_MARKER).write_text(info["finished_at"])
        logger.warning("✅ Finished run: %s", out_dir)
    return scores_df


def execute(cfg) -> pd.DataFrame:
    """Run one experiment cell end-to-end; return the evaluated prompt/score table.

    Writes the per-run output to ``cfg.out_dir``, which Hydra sets per run/cell: per-step trace, final
    scores, name/status/timestamps and a ``.finished`` marker. If that directory already holds
    ``.finished`` and ``cfg.skip_completed`` is set, the run is skipped and its scores returned.

    Args:
        cfg: A composed experiment config (see ``conf/config.yaml`` for the schema).

    Returns:
        pd.DataFrame: Columns ``prompt`` and ``score``, best first.
    """
    full_df = instantiate(cfg.task.df)  # the dataset: a _target_ that returns a df (or a df-like)
    train_df, test_df = train_test_split(full_df, test_frac=cfg.test_frac, seed=cfg.random_seed)
    train_task = instantiate(cfg.task, df=train_df)  # df kwarg overrides the nested df config
    test_task = instantiate(cfg.task, df=test_df) if cfg.test_frac > 0 else train_task
    return _run(cfg, train_task, test_task, out_dir=cfg.out_dir)


def optimize(
    data: Union[pd.DataFrame, str, Path],
    task_description: str,
    task_type: str = "classification",
    *,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    x_column: str = "x",
    y_column: str = "y",
    n_steps: Optional[int] = None,
    optimizer: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Optimize prompts for a classification task, on the shipped Hydra defaults.

    A lightweight entry point that composes ``conf/config.yaml`` (optimizer CAPO, predictor
    MarkerBasedPredictor, 10 steps by default) and asks only for what the library cannot know:
    your data, an LLM and its credentials, the task description, and the task type.

    Only classification is supported here. For judge or reward tasks, build the components
    directly and call ``optimizer.optimize()`` (see the README quickstart).

    Args:
        data (Union[pd.DataFrame, str, Path]): A DataFrame, or a path/URL ``pandas.read_csv`` accepts.
        task_description (str): Description of the task, passed to the meta-LLM.
        task_type (str): Only ``"classification"`` is supported; anything else raises.
        model_id (Optional[str]): Overrides the default LLM's model id.
        api_key (Optional[str]): API key for the LLM. If omitted, resolved from the environment
            (e.g. ``OPENAI_API_KEY``) by the underlying OpenAI client.
        api_url (Optional[str]): API base URL for the LLM.
        x_column (str): Input column name in ``data``.
        y_column (str): Label column name in ``data``.
        n_steps (Optional[int]): Overrides the default number of optimization steps.
        optimizer (Optional[str]): Overrides the default optimizer group (e.g. ``"opro"``).
        output_dir (Optional[Union[str, Path]]): If set, also writes the run contract (per-step
            trace, final scores, runinfo, restart marker) here, same as the CLI. If omitted
            (the default), nothing is written to disk.
        overrides (Optional[List[str]]): Extra Hydra overrides, e.g. ``["predictor=first_occurrence"]``.

    Returns:
        pd.DataFrame: Columns ``prompt`` and ``score``, best first.
    """
    if task_type != "classification":
        raise NotImplementedError(
            "optimize() supports classification only. For judge or reward tasks, build the "
            "components directly and call optimizer.optimize() (see the README quickstart)."
        )

    overrides_list = list(overrides or [])
    if optimizer is not None:
        overrides_list.append(f"optimizer={optimizer}")
    if n_steps is not None:
        overrides_list.append(f"n_steps={n_steps}")
    cfg = _compose(overrides_list)

    OmegaConf.set_struct(cfg, False)
    if model_id is not None:
        cfg.llm.model_id = model_id
    if api_key is not None:
        cfg.llm.api_key = api_key
    if api_url is not None:
        cfg.llm.api_url = api_url
    if output_dir is not None and OmegaConf.is_missing(cfg, "name"):
        cfg.name = Path(output_dir).name

    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    train_df, test_df = train_test_split(df, test_frac=cfg.test_frac, seed=cfg.random_seed)
    train_task = ClassificationTask(train_df, task_description=task_description, x_column=x_column, y_column=y_column)
    test_task = (
        ClassificationTask(test_df, task_description=task_description, x_column=x_column, y_column=y_column)
        if cfg.test_frac > 0
        else train_task
    )
    return _run(cfg, train_task, test_task, out_dir=output_dir)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    """Execute the cell Hydra composed, in the run dir it created (``cfg.out_dir``)."""
    execute(cfg)


if __name__ == "__main__":
    main()
