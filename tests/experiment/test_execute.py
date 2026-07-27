"""End-to-end tests of execute(): instantiate wiring + output contract (no real LLM).

The LLM is swapped for MockLLM by replacing cfg.llm, so instantiate builds it like any component.
"""

import json
from unittest.mock import patch

import pandas as pd

from tests.mocks.mock_llm import MockLLM

from promptolution.experiment import execute
from promptolution.tasks.classification_tasks import ClassificationTask


def test_execute_wiring(cfg_with_mock_llm, cell_prompts):
    """execute() builds the right components, splits train/test, and evaluates on the held-out task."""
    cfg = cfg_with_mock_llm(["name=t", "task=demo", "optimizer=evopromptga", "n_steps=2", "test_frac=0.25"])
    with patch("promptolution.experiment.launch.evaluate_prompts") as mock_evaluate_prompts:
        mock_evaluate_prompts.return_value = pd.DataFrame({"prompt": ["p"], "score": [1.0]})
        execute(cfg)
    prompts, test_task, predictor = mock_evaluate_prompts.call_args[0]
    assert len(prompts) == len(cell_prompts)
    assert isinstance(test_task, ClassificationTask)
    assert len(test_task.df) == 2  # 25% of the demo task's 8 rows held out
    assert isinstance(predictor.llm, MockLLM)


def test_execute_no_split_evaluates_on_train(cfg_with_mock_llm):
    """test_frac=0 -> no held-out split; evaluation runs on the train task."""
    cfg = cfg_with_mock_llm(["name=t", "task=demo", "optimizer=evopromptga", "n_steps=2", "test_frac=0"])
    with patch("promptolution.experiment.launch.evaluate_prompts") as mock_evaluate_prompts:
        mock_evaluate_prompts.return_value = pd.DataFrame({"prompt": ["p"], "score": [1.0]})
        execute(cfg)
    _, task, _ = mock_evaluate_prompts.call_args[0]
    assert len(task.df) == 8  # the full demo dataset


def test_execute_end_to_end(cfg_with_mock_llm, tmp_path):
    cfg = cfg_with_mock_llm(["name=t", "task=demo", "optimizer=evopromptga", "n_steps=2", "test_frac=0.25"])
    out = execute(cfg)
    assert isinstance(out, pd.DataFrame) and "score" in out.columns
    for f in (".finished", "prompt_scores.parquet", "step_results.parquet", "runinfo.json"):
        assert (tmp_path / f).exists(), f"missing {f}"
    info = json.loads((tmp_path / "runinfo.json").read_text())
    assert info["status"] == "finished" and info["name"] == "t"
