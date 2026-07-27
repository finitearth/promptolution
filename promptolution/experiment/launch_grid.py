"""Build and run one experiment cell via Hydra `instantiate`."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate

from promptolution.utils.callbacks import FileOutputCallback
from promptolution.utils.evaluation import score_prompts, train_test_split
from promptolution.utils.logging import get_logger

logger = get_logger(__name__)

FINISHED_MARKER = ".finished"


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
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.skip_completed and (out_dir / FINISHED_MARKER).exists():
        logger.warning("⏭️  Skipping finished run: %s", out_dir)
        return pd.read_parquet(out_dir / "prompt_scores.parquet")

    llm = instantiate(cfg.llm)
    meta_llm = instantiate(cfg.meta_llm) if cfg.meta_llm else llm  # null in config -> share the llm
    predictor = instantiate(cfg.predictor, llm=llm)

    full_df = instantiate(cfg.task.df)  # the dataset: a _target_ that returns a df (or a df-like)
    train_df, test_df = train_test_split(full_df, test_frac=cfg.test_frac, seed=cfg.random_seed)
    train_task = instantiate(cfg.task, df=train_df)  # df kwarg overrides the nested df config
    test_task = instantiate(cfg.task, df=test_df) if cfg.test_frac > 0 else train_task

    optimizer = instantiate(cfg.optimizer, predictor=predictor, meta_llm=meta_llm, task=train_task)
    optimizer.callbacks.append(FileOutputCallback(dir=str(out_dir)))

    info = {"name": cfg.name, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    (out_dir / "runinfo.json").write_text(json.dumps(info, indent=2))
    try:
        logger.warning("🔥 Starting optimization...")
        prompts = optimizer.optimize(n_steps=cfg.n_steps)
        scores_df = score_prompts(prompts, test_task, predictor)
    except Exception as e:
        info.update(status="failed", finished_at=datetime.now(timezone.utc).isoformat(), error=str(e))
        (out_dir / "runinfo.json").write_text(json.dumps(info, indent=2))
        raise

    scores_df.to_parquet(out_dir / "prompt_scores.parquet", index=False)
    info.update(status="finished", finished_at=datetime.now(timezone.utc).isoformat())
    (out_dir / "runinfo.json").write_text(json.dumps(info, indent=2))
    (out_dir / FINISHED_MARKER).write_text(info["finished_at"])
    logger.warning("✅ Finished run: %s", out_dir)
    return scores_df
