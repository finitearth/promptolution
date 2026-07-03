"""Hydra entry point for promptolution experiment grids.

Design: the CLI ``@hydra.main`` ``main`` is the grid entry — so Hydra's
multirun (``-m``) and the submitit SLURM launcher work exactly as intended — but the per-cell work is
factored into a plain, directly-testable ``execute(cfg)`` seam, with ``compose_experiment`` for
notebook/programmatic use. The end-user library API ``promptolution.helpers.run_experiment`` stays
Hydra-free; Hydra lives only in this module (the optional ``[experiments]`` extra).

Usage:
    python -m promptolution.experiments.run optimizer=capo dataset=agnews          # single run
    python -m promptolution.experiments.run -m optimizer=capo,opro random_seed=42,43  # grid
    python -m promptolution.experiments.run -m hydra/launcher=submitit_slurm ...    # SLURM array job
    python -m promptolution.experiments.run smoke=true                             # offline, no LLM
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from promptolution.experiments import results as results_mod
from promptolution.experiments.bridge import build_experiment_config
from promptolution.utils.logging import get_logger

logger = get_logger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parent / "conf")


def execute(cfg, out_dir: Optional[Path] = None) -> Path:
    """Run one grid cell end-to-end; return its output directory.

    Steps: resolve the output dir → (optionally skip if already finished) → load the dataset bundle
    (``instantiate(cfg.dataset)``) → bridge to an ``ExperimentConfig`` → run via the existing
    ``run_experiment`` (attaching the per-step results callback) → persist outputs + markers.

    ``out_dir`` defaults to Hydra's per-run output dir; it can be passed explicitly for programmatic
    use / tests (so ``execute`` is testable without a Hydra runtime).
    """
    from hydra.utils import instantiate

    if out_dir is None:
        from hydra.core.hydra_config import HydraConfig

        out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("skip_completed", True) and results_mod.is_finished(out_dir):
        logger.warning("⏭️  Skipping already-finished run: %s", out_dir)
        return out_dir

    results_mod.write_runinfo(out_dir, cfg, status="running")
    results_mod.write_resolved_config(out_dir, cfg)
    try:
        bundle = instantiate(cfg.dataset)
        exp = build_experiment_config(cfg, bundle)

        if cfg.get("smoke", False):
            result = _smoke_run(bundle, exp, out_dir)
        else:
            from promptolution.helpers import run_experiment

            callbacks: List = [results_mod.StepResultsCallback(out_dir)]
            result = run_experiment(bundle.df, exp, callbacks=callbacks)

        result.to_parquet(out_dir / "prompt_scores.parquet", index=False)
        results_mod.write_runinfo(out_dir, cfg, status="finished")
        results_mod.mark_finished(out_dir)
        logger.warning("✅ Finished run: %s", out_dir)
    except Exception as e:  # noqa: BLE001 - record failure, then re-raise for the launcher
        results_mod.write_runinfo(out_dir, cfg, status="failed", error=str(e))
        logger.error("⛔ Run failed: %s", out_dir, exc_info=e)
        raise
    return out_dir


def _smoke_run(bundle, exp, out_dir: Path) -> pd.DataFrame:
    """Offline stand-in for ``run_experiment`` — exercises the full plumbing without an LLM.

    Writes a minimal ``step_results.parquet`` (same schema as ``FileOutputCallback``) and returns a
    prompt/score table, so single runs, grids, and the SLURM launcher can be validated with no
    API/GPU. Selected via ``smoke=true``.
    """
    prompts = [str(p) for p in (getattr(exp, "prompts", None) or ["mock prompt"])]
    scores = [round(1.0 - 0.1 * i, 3) for i in range(len(prompts))]
    pd.DataFrame(
        {
            "step": [1] * len(prompts),
            "score": scores,
            "prompt": prompts,
            "input_tokens": [0] * len(prompts),
            "output_tokens": [0] * len(prompts),
            "time": [0.0] * len(prompts),
        }
    ).to_parquet(out_dir / "step_results.parquet", index=False)
    return pd.DataFrame({"prompt": prompts, "score": scores}).sort_values(
        "score", ascending=False, ignore_index=True
    )


def compose_experiment(overrides: Optional[List[str]] = None, config_name: str = "config"):
    """Compose a single experiment config programmatically (notebooks/tests), without ``@hydra.main``.

    Note: Hydra *sweeps* (multirun) are driven by the CLI ``-m`` path, not by ``compose`` — use the
    ``python -m promptolution.experiments.run -m ...`` entry for grids.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides or [])


def _build_main():
    import hydra

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(cfg) -> None:
        execute(cfg)

    return main


main = _build_main()


if __name__ == "__main__":
    main()
