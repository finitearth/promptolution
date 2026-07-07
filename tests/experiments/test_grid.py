"""Slice 3: restart (name-keyed dir, skip finished) + grid sources (run_grid, YAML grid config)."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("hydra")
from omegaconf import OmegaConf

from promptolution.experiments.launch import compose_experiment, execute, run_grid

_PROMPTS = ["Classify the sentiment. <final_answer></final_answer>", "Positive, negative or neutral?"]


def _cfg_with_mock_llm(overrides):
    cfg = compose_experiment(overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.llm = {
        "_target_": "tests.mocks.mock_llm.MockLLM",
        "predetermined_responses": ["<final_answer>positive</final_answer>"] * 200,
    }
    cfg.optimizer.initial_prompts = list(_PROMPTS)
    return cfg


def test_restart_skips_finished(tmp_path):
    """A second run into the same folder skips the finished cell (does not re-execute)."""
    cfg = _cfg_with_mock_llm(["task=dummy", "optimizer=evopromptga", "n_steps=2", "test_frac=0.25"])
    execute(cfg, out_dir=tmp_path)
    assert (tmp_path / ".finished").exists()
    (tmp_path / "step_results.parquet").unlink()             # remove an output
    execute(cfg, out_dir=tmp_path)                            # skip_completed default -> skip
    assert not (tmp_path / "step_results.parquet").exists()   # not regenerated => skipped


def test_run_grid_expands_and_is_resumable(tmp_path):
    """run_grid builds the cartesian product, self-labels the cell dirs, and skips finished on rerun."""
    ran = []

    def fake_execute(cfg, out_dir=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if (Path(out_dir) / ".finished").exists():
            return pd.DataFrame({"prompt": ["p"], "score": [0.0]})  # already done -> skip
        (Path(out_dir) / ".finished").write_text("")
        ran.append((cfg.optimizer._target_.split(".")[-1], int(cfg.random_seed)))
        return pd.DataFrame({"prompt": ["p"], "score": [1.0]})

    grid = {"optimizer": ["capo", "opro"], "random_seed": [42, 43]}
    with patch("promptolution.experiments.launch.execute", side_effect=fake_execute):
        res = run_grid(grid, name="g", output_root=str(tmp_path))
        assert len(res) == 4 and len(ran) == 4  # 2 x 2 product, all executed
        assert set(res) == {
            "optimizer=capo,random_seed=42", "optimizer=capo,random_seed=43",
            "optimizer=opro,random_seed=42", "optimizer=opro,random_seed=43",
        }
        assert (tmp_path / "g" / "optimizer=capo,random_seed=42" / ".finished").exists()
        ran.clear()
        run_grid(grid, name="g", output_root=str(tmp_path))  # rerun -> all skipped
        assert ran == []


def test_yaml_grid_config_composes():
    """A grid defined in a file (grid_example.yaml) inherits the base config + names the experiment."""
    cfg = compose_experiment(config_name="grid_example")
    assert cfg.name == "example_grid"
    assert cfg.task.df.path == "SetFit/ag_news"  # task=agnews override applied on top of base config
