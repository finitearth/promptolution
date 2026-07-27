"""Config-driven (Hydra) experiments for promptolution — a single run or a whole grid.

:func:`execute` builds the components of one experiment cell from its config via
`hydra.utils.instantiate` (``llm -> predictor(llm) -> task(df) -> optimizer``), optimizes, evaluates,
and writes the per-run output contract (results, runinfo, restart marker) — see `README.md`.

Usage:
    python -m promptolution.experiment_grid name=my_run optimizer=capo task=agnews llm=api  # one run
    python -m promptolution.experiment_grid -m name=bench optimizer=capo,opro random_seed=42,43  # a grid
"""

from promptolution.experiment_grid.launch_grid import execute

__all__ = ["execute"]
