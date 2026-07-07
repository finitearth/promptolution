"""Hydra-based experiment gridding for promptolution (deep instantiate).

Optional add-on (`pip install promptolution[experiments]`). Define a grid of experiments in `conf/`
(each component group is a `_target_` + params), run it locally or as a SLURM array job; every cell
builds the components via `instantiate` and executes them through `promptolution.runner.run`.

Entry points:
- CLI:          `python -m promptolution.experiments.run [overrides...] [-m for grids]`
- Programmatic: :func:`compose_experiment` + :func:`execute`
"""

__all__ = ["execute", "compose_experiment"]


def __getattr__(name: str):  # lazy — avoids importing Hydra unless actually used
    if name in __all__:
        from promptolution.experiments import launch

        return getattr(launch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
