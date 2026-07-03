"""End-to-end tests of the experiments module through its interface (execute / run_experiment).

Per dependency category (DEEPENING): the offline ``smoke`` path needs no external deps; the
integration test doubles the true-external optimization via ``tests/mocks`` + patched ``get_*``.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("hydra")

from tests.mocks.mock_llm import MockLLM
from tests.mocks.mock_predictor import MockPredictor

from promptolution.experiments.run import compose_experiment, execute
from promptolution.helpers import get_optimizer
from promptolution.tasks.base_task import EvalResult
from promptolution.utils import ExperimentConfig
from promptolution.utils.prompt import Prompt


def test_execute_smoke_writes_output_contract(tmp_path):
    cfg = compose_experiment(overrides=["smoke=true"])
    out = execute(cfg, out_dir=tmp_path)
    assert out == tmp_path
    for name in ("step_results.parquet", "prompt_scores.parquet", "runinfo.json", "experiment_config.yaml", ".finished"):
        assert (tmp_path / name).exists(), f"missing {name}"
    info = json.loads((tmp_path / "runinfo.json").read_text())
    assert info["status"] == "finished"


def test_skip_completed(tmp_path):
    cfg = compose_experiment(overrides=["smoke=true"])
    execute(cfg, out_dir=tmp_path)
    (tmp_path / "step_results.parquet").unlink()  # remove an output
    execute(cfg, out_dir=tmp_path)  # should skip (skip_completed defaults True)
    assert not (tmp_path / "step_results.parquet").exists()  # not regenerated => skipped


def test_get_optimizer_forwards_callbacks():
    """Core change: callbacks now reach the optimizer (so per-run results can be written)."""
    cb = MagicMock()
    cfg = ExperimentConfig(prompts=[Prompt("p1"), Prompt("p2")], task_description="desc")
    opt = get_optimizer(
        MockPredictor(llm=MockLLM()), MockLLM(), MagicMock(task_type="classification"),
        optimizer="evopromptga", config=cfg, callbacks=[cb],
    )
    assert cb in opt.callbacks


def test_execute_integration_real_run_experiment(tmp_path):
    """Drive the real run_experiment path (bridge -> helpers) with mocked components."""
    cfg = compose_experiment()  # smoke=false; uses dummy dataset (has initial_prompts)
    eval_result = EvalResult(
        scores=np.array([[0.9], [0.8]], dtype=float),
        agg_scores=np.array([0.9, 0.8], dtype=float),
        sequences=np.array([["s1"], ["s2"]], dtype=object),
        input_tokens=np.array([[10.0], [10.0]], dtype=float),
        output_tokens=np.array([[5.0], [5.0]], dtype=float),
        agg_input_tokens=np.array([10.0, 10.0], dtype=float),
        agg_output_tokens=np.array([5.0, 5.0], dtype=float),
    )
    with patch("promptolution.helpers.get_llm", return_value=MockLLM()), patch(
        "promptolution.helpers.get_predictor", return_value=MockPredictor()
    ), patch("promptolution.helpers.get_task") as mock_get_task, patch(
        "promptolution.helpers.get_optimizer"
    ) as mock_get_optimizer:
        mock_task = MagicMock()
        mock_task.evaluate.return_value = eval_result
        mock_get_task.return_value = mock_task
        mock_opt = MagicMock()
        mock_opt.optimize.return_value = [Prompt("a"), Prompt("b")]
        mock_get_optimizer.return_value = mock_opt

        out = execute(cfg, out_dir=tmp_path)

    assert (out / "prompt_scores.parquet").exists()
    assert json.loads((out / "runinfo.json").read_text())["status"] == "finished"
    mock_opt.optimize.assert_called_once()
