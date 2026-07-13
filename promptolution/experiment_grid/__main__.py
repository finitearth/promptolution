"""CLI entry: ``python -m promptolution.experiment_grid [overrides...]`` (``-m`` for grids)."""

from hydra.core.hydra_config import HydraConfig
from pathlib import Path

import hydra

from promptolution.experiment_grid import execute


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    """Execute the cell in the run dir Hydra composed and created."""
    execute(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
