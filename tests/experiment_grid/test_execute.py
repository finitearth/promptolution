"""End-to-end tests of execute(): instantiate wiring + output contract (no real LLM).

The LLM is swapped for MockLLM by replacing cfg.llm, so instantiate builds it like any component.
"""

import json
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

from tests.mocks.mock_llm import MockLLM

from promptolution.experiment_grid import compose_experiment, execute
from promptolution.tasks.classification_tasks import ClassificationTask

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
    """execute() builds the right components, splits train/test, and evaluates on the held-out task."""
    cfg = _cfg_with_mock_llm(["name=t", "task=dummy", "optimizer=evopromptga", "n_steps=2", "test_frac=0.25"])
    with patch("promptolution.experiment_grid.launch_grid.evaluate") as mock_evaluate:
        mock_evaluate.return_value = pd.DataFrame({"prompt": ["p"], "score": [1.0]})
        execute(cfg, out_dir=tmp_path)
    prompts, test_task, predictor = mock_evaluate.call_args[0]
    assert len(prompts) == len(_PROMPTS)
    assert isinstance(test_task, ClassificationTask)
    assert len(test_task.df) == 2  # 25% of the dummy task's 8 rows held out
    assert isinstance(predictor.llm, MockLLM)


def test_execute_no_split_evaluates_on_train(tmp_path):
    """test_frac=0 -> no held-out split; evaluation runs on the train task."""
    cfg = _cfg_with_mock_llm(["name=t", "task=dummy", "optimizer=evopromptga", "n_steps=2", "test_frac=0"])
    with patch("promptolution.experiment_grid.launch_grid.evaluate") as mock_evaluate:
        mock_evaluate.return_value = pd.DataFrame({"prompt": ["p"], "score": [1.0]})
        execute(cfg, out_dir=tmp_path)
    _, task, _ = mock_evaluate.call_args[0]
    assert len(task.df) == 8  # the full dummy dataset


def test_execute_end_to_end(tmp_path):
    cfg = _cfg_with_mock_llm(["name=t", "task=dummy", "optimizer=evopromptga", "n_steps=2", "test_frac=0.25"])
    out = execute(cfg, out_dir=tmp_path)
    assert isinstance(out, pd.DataFrame) and "score" in out.columns
    for f in (".finished", "prompt_scores.parquet", "step_results.parquet", "runinfo.json"):
        assert (tmp_path / f).exists(), f"missing {f}"
    info = json.loads((tmp_path / "runinfo.json").read_text())
    assert info["status"] == "finished" and info["name"] == "t"
