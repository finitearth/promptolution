"""Config-driven (Hydra) experiments for promptolution — a single run or a whole grid.

:func:`execute` builds the components of one experiment cell from its config via
`hydra.utils.instantiate` (``llm -> predictor(llm) -> task(df) -> optimizer``), optimizes, evaluates,
and writes the per-run output contract (results, runinfo, restart marker) — see `README.md`.

For a lightweight, in-memory alternative that skips the CLI and YAML entirely, see
:func:`promptolution.optimize` (classification only).

Usage:
    promptolution-experiment name=my_run optimizer=capo task=agnews llm=api  # one run
    promptolution-experiment -m name=bench optimizer=capo,opro random_seed=42,43  # a grid

    # no install? the same CLI also runs as a module:
    python -m promptolution.experiment.launch name=my_run optimizer=capo task=agnews llm=api
"""

from promptolution.experiment.launch import build_components, execute, optimize

__all__ = ["execute", "optimize", "build_components"]
