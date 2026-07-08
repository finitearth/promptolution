"""End-to-end tests of execute(): instantiate wiring + runner integration (no real LLM).

The LLM is swapped for MockLLM by replacing cfg.llm, so instantiate builds it like any component.
"""

import json
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("hydra")
from omegaconf import OmegaConf  # noqa: E402

from tests.mocks.mock_llm import MockLLM  # noqa: E402

from promptolution.experiments.launch import compose_experiment, execute  # noqa: E402
from promptolution.tasks.classification_tasks import ClassificationTask  # noqa: E402

_PROMPTS = ["Classify the sentiment. <final_answer></final_answer>", "Positive, negative or neutral?"]


def _cfg_with_mock_llm(overrides):
    cfg = compose_experiment(overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.llm = {
        "_target_": "tests.mocks.mock_llm.MockLLM",
        "predetermined_responses": ["<final_answer>positive</final_answer>"] * 200,
    }
    cfg.optimizer.initial_prompts = list(_PROMPTS)  # avoid the auto-generate path in tests
    return cfg


def test_execute_wiring(tmp_path):
    """execute() builds the right components + splits train/test, and hands them to runner.run."""
    cfg = _cfg_with_mock_llm(["task=dummy", "optimizer=evopromptga", "n_steps=2", "test_frac=0.25"])
    with patch("promptolution.experiments.launch.run") as mock_run:
        mock_run.return_value = pd.DataFrame({"prompt": ["p"], "score": [1.0]})
        execute(cfg, out_dir=tmp_path)
    (optimizer,), kw = mock_run.call_args
    assert isinstance(optimizer.task, ClassificationTask)
    assert isinstance(optimizer.predictor.llm, MockLLM)
    assert kw["test_task"] is not None  # test_frac>0 -> a held-out Task was built
    assert len(optimizer.task.df) < 8  # train split is smaller than the full 8 rows


def test_execute_end_to_end(tmp_path):
    cfg = _cfg_with_mock_llm(["task=dummy", "optimizer=evopromptga", "n_steps=2", "test_frac=0.25"])
    out = execute(cfg, out_dir=tmp_path)
    assert isinstance(out, pd.DataFrame) and "score" in out.columns
    for f in (".finished", "prompt_scores.parquet", "step_results.parquet", "runinfo.json"):
        assert (tmp_path / f).exists(), f"missing {f}"
    assert json.loads((tmp_path / "runinfo.json").read_text())["status"] == "finished"
