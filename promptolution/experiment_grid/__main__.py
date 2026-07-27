"""CLI entry: ``python -m promptolution.experiment_grid [overrides...]`` (``-m`` for grids)."""

import hydra

from promptolution.experiment_grid.launch_grid import execute


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    """Execute the cell Hydra composed, in the run dir it created (``cfg.out_dir``)."""
    execute(cfg)


if __name__ == "__main__":
    main()
