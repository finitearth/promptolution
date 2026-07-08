"""End-to-end experiment runner for promptolution.

Components-as-interface: build your `llm` / `task` / `predictor` / `optimizer` directly, then hand the
optimizer to :func:`run` to optimize, (optionally) evaluate on a held-out task, and write the per-run
output contract. :func:`train_test_split` helps carve out a held-out set.

This module is Hydra-free; the grid layer (`promptolution.experiments`) builds the same components from
YAML via `instantiate` and calls the same :func:`run`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from promptolution.utils.callbacks import FileOutputCallback
from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.optimizers.base_optimizer import BaseOptimizer
    from promptolution.tasks.base_task import BaseTask

logger = get_logger(__name__)

FINISHED_MARKER = ".finished"


def train_test_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into ``(train_df, test_df)``.

    Accepts anything df-like (e.g. a HuggingFace ``Dataset`` straight from ``datasets.load_dataset``),
    normalizing to pandas first — duck-typed, so ``datasets`` stays an optional dependency.
    """
    to_pandas = getattr(df, "to_pandas", None)
    if to_pandas is not None:
        df = to_pandas()
    test_df = df.sample(frac=test_frac, random_state=seed)
    train_df = df.drop(test_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run(
    optimizer: "BaseOptimizer",
    n_steps: int,
    *,
    test_task: Optional["BaseTask"] = None,
    output_dir: Optional[Union[str, Path]] = None,
    name: Optional[str] = None,
    skip_completed: bool = True,
) -> pd.DataFrame:
    """Optimize with a fully-constructed optimizer, then evaluate and (optionally) persist results.

    Args:
        optimizer: A constructed optimizer (its train ``Task``, predictor and meta LLM already set).
        n_steps: Number of optimization steps.
        test_task: Optional held-out ``Task``; if given, the final prompts are scored on it, otherwise
            they're scored on the optimizer's own (train) task.
        output_dir: If set, write ``step_results.parquet`` / ``prompt_scores.parquet`` /
            ``runinfo.json`` / ``.finished`` here.
        name: Human-readable experiment name (recorded in ``runinfo.json``).
        skip_completed: If ``output_dir`` already holds a ``.finished`` marker, skip and return its
            ``prompt_scores.parquet`` (restart / resume).

    Returns:
        DataFrame of ``prompt`` and ``score``, sorted best-first.
    """
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if skip_completed and (out / FINISHED_MARKER).exists():
            logger.warning("⏭️  Skipping finished run: %s", out)
            return pd.read_parquet(out / "prompt_scores.parquet")
        optimizer.callbacks = list(optimizer.callbacks) + [FileOutputCallback(dir=str(out))]
        _write_runinfo(out, name, status="running")

    try:
        logger.warning("🔥 Starting optimization...")
        prompts = optimizer.optimize(n_steps=n_steps)
        scores_df = _evaluate(prompts, optimizer, test_task)
        if out is not None:
            scores_df.to_parquet(out / "prompt_scores.parquet", index=False)
            _write_runinfo(out, name, status="finished")
            (out / FINISHED_MARKER).write_text(datetime.now(timezone.utc).isoformat())
            logger.warning("✅ Finished run: %s", out)
    except Exception as e:  # noqa: BLE001 - record failure, then re-raise
        if out is not None:
            _write_runinfo(out, name, status="failed", error=str(e))
        raise
    return scores_df


def _evaluate(prompts: List, optimizer: "BaseOptimizer", test_task: Optional["BaseTask"]) -> pd.DataFrame:
    """Score ``prompts`` on ``test_task`` (or the optimizer's own task) and return a sorted table."""
    task = test_task if test_task is not None else optimizer.task
    if isinstance(prompts[0], str):
        str_prompts = list(prompts)
        prompt_objs = [Prompt(p) for p in str_prompts]
    else:
        str_prompts = [p.construct_prompt() for p in prompts]
        prompt_objs = list(prompts)
    logger.warning("📊 Starting evaluation...")
    results = task.evaluate(prompt_objs, optimizer.predictor, eval_strategy="full")
    return pd.DataFrame({"prompt": str_prompts, "score": results.agg_scores.tolist()}).sort_values(
        "score", ascending=False, ignore_index=True
    )


def _write_runinfo(out: Path, name: Optional[str], status: str, error: Optional[str] = None) -> None:
    info_path = out / "runinfo.json"
    existing = json.loads(info_path.read_text()) if info_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    info = {
        **existing,
        "name": name,
        "status": status,
        "started_at": existing.get("started_at", now),
        "updated_at": now,
        "error": error,
    }
    info_path.write_text(json.dumps(info, indent=2))
