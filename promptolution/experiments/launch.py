"""Config-driven (Hydra) entry point for promptolution runs — a single run or a whole grid.

This is the CLI/config layer over the Hydra-free core runner (`promptolution.runner.run`). `execute(cfg)`
builds the components directly from config via `hydra.utils.instantiate`
(`llm -> predictor(llm) -> task(df) -> optimizer(predictor, meta_llm, task)`), splits the task's data
into train/test, and delegates to `runner.run`. No `ExperimentConfig`, no bridge, no string dispatch.

It lives in the optional `experiments` package because it requires Hydra (the `[experiments]` extra)
and is cohesive with `conf/`. A plain Python run needs no Hydra — just construct the components and call
`promptolution.runner.run` directly.

Usage:
    python -m promptolution.experiments.launch optimizer=capo task=agnews llm=api   # a single run
    python -m promptolution.experiments.launch -m optimizer=capo,opro random_seed=42,43   # a grid
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import hydra

from promptolution.runner import run, train_test_split
from promptolution.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = get_logger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parent / "conf")


def execute(cfg, out_dir: Optional[Path] = None) -> "pd.DataFrame":
    """Run one experiment cell end-to-end; return the evaluated prompt/score table.

    Steps: instantiate `llm`, `predictor(llm)`, the dataset `df`, split it, build train/test `Task`s
    and the `optimizer`, then delegate to `runner.run`. `out_dir` defaults to Hydra's per-run dir but
    can be passed explicitly (programmatic use / tests).
    """
    from hydra.utils import instantiate

    if out_dir is None:
        from hydra.core.hydra_config import HydraConfig

        out_dir = HydraConfig.get().runtime.output_dir
    out_dir = Path(out_dir)

    llm = instantiate(cfg.llm)
    meta_llm = instantiate(cfg.meta_llm) if cfg.get("meta_llm") is not None else llm
    predictor = instantiate(cfg.predictor, llm=llm)

    full_df = instantiate(cfg.task.df)  # the dataset: a _target_ that returns a df (or a df-like)
    test_frac = float(cfg.get("test_frac", 0.2))
    if test_frac > 0:
        train_df, test_df = train_test_split(full_df, test_frac=test_frac, seed=int(cfg.get("random_seed", 42)))
    else:
        train_df, test_df = full_df, None

    train_task = instantiate(cfg.task, df=train_df)  # df kwarg overrides the nested df config
    test_task = instantiate(cfg.task, df=test_df) if test_df is not None else None

    optimizer = instantiate(cfg.optimizer, predictor=predictor, meta_llm=meta_llm, task=train_task)

    return run(
        optimizer,
        n_steps=int(cfg.n_steps),
        test_task=test_task,
        output_dir=str(out_dir),
        name=cfg.get("name"),
        skip_completed=bool(cfg.get("skip_completed", True)),
    )


def compose_experiment(overrides: Optional[List[str]] = None, config_name: str = "config"):
    """Compose an experiment config programmatically (notebooks/tests), without `@hydra.main`.

    Note: Hydra *sweeps* (multirun) run through the CLI `-m`, not `compose`; use `run_grid` (Slice 3)
    or the CLI for grids.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides or [])


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    """CLI entry: Hydra composes `cfg` + creates the run dir, then we execute the cell."""
    execute(cfg)


if __name__ == "__main__":
    main()
