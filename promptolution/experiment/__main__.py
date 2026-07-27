"""CLI entry: ``python -m promptolution.experiment [overrides...]`` (``-m`` for grids)."""

import hydra

from promptolution.experiment.launch_grid import execute


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    """Execute the cell Hydra composed, in the run dir it created (``cfg.out_dir``)."""
    execute(cfg)


if __name__ == "__main__":
    main()
