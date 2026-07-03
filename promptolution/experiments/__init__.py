"""Hydra-based experiment gridding for promptolution.

Define a grid of experiments once (in ``conf/``), run it locally or as a single SLURM array job, and
get per-run outputs + restart for free. This is an **optional** layer (``pip install
promptolution[experiments]``); the base library and ``run_experiment`` stay Hydra-free.

Entry points:
- CLI:        ``python -m promptolution.experiments.run [overrides...] [-m for grids]``
- Programmatic: :func:`compose_experiment` + :func:`execute`

The dataset loaders (pandas-only) are importable without Hydra; ``execute``/``compose_experiment``
pull in Hydra lazily.
"""

from promptolution.experiments.datasets import DatasetBundle, load_hf, load_inline

__all__ = [
    "DatasetBundle",
    "load_hf",
    "load_inline",
    "execute",
    "compose_experiment",
    "build_experiment_config",
]


def __getattr__(name: str):  # lazy — avoids importing Hydra/omegaconf unless actually used
    if name in ("execute", "compose_experiment"):
        from promptolution.experiments import run

        return getattr(run, name)
    if name == "build_experiment_config":
        from promptolution.experiments.bridge import build_experiment_config

        return build_experiment_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
