"""Tests for promptolution.optimize() and promptolution.evaluate(): the lightweight entry points."""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from tests.mocks.mock_llm import MockLLM

from promptolution import evaluate, optimize
from promptolution.utils.prompt import Prompt

_DF = pd.DataFrame(
    {
        "x": ["I love it", "Terrible experience", "It is okay", "Absolutely amazing", "Awful and broken", "Not bad"],
        "y": ["positive", "negative", "neutral", "positive", "negative", "positive"],
    }
)
_PROMPTS = ["Classify the sentiment. <final_answer></final_answer>", "Positive, negative or neutral?"]


def _mock_llm():
    return MockLLM(predetermined_responses=["<final_answer>positive</final_answer>"] * 200)


def _optimize(**kwargs):
    """Run optimize() with the API LLM swapped for a MockLLM."""
    with patch("promptolution.utils.optimize.APILLM") as mock_apillm:
        mock_apillm.return_value = _mock_llm()
        return optimize(
            _DF,
            task_description="Classify the sentiment.",
            optimizer="evopromptga",
            n_steps=2,
            initial_prompts=list(_PROMPTS),
            **kwargs,
        )


def test_optimize_returns_prompts_and_writes_nothing(tmp_path, monkeypatch):
    """optimize() returns prompts only; evaluation is a separate call."""
    monkeypatch.chdir(tmp_path)
    prompts = _optimize()
    assert isinstance(prompts, list) and len(prompts) == len(_PROMPTS)
    assert all(isinstance(p, Prompt) for p in prompts)
    assert list(tmp_path.iterdir()) == []  # no output_dir -> nothing written to disk


def test_optimize_output_dir_writes_trace_and_runinfo(tmp_path):
    """With output_dir, the step trace and run info are written; scores are not, evaluate() does that."""
    out_dir = tmp_path / "run"
    _optimize(output_dir=out_dir)
    for f in ("step_results.parquet", "runinfo.json"):
        assert (out_dir / f).exists(), f"missing {f}"
    assert not (out_dir / "prompt_scores.parquet").exists()  # scoring is no longer optimize()'s job
    info = json.loads((out_dir / "runinfo.json").read_text())
    assert info["status"] == "finished" and info["name"] == "run"  # taken from the output dir's name


def test_optimize_generates_initial_prompts_from_task_description():
    """initial_prompts is optional; they are generated from the task description instead."""
    with patch("promptolution.optimizers.base_optimizer.create_prompts_from_task_description") as mock_create, patch(
        "promptolution.utils.optimize.APILLM"
    ) as mock_apillm:
        mock_create.return_value = list(_PROMPTS)
        mock_apillm.return_value = _mock_llm()
        prompts = optimize(_DF, task_description="Classify the sentiment.", optimizer="evopromptga", n_steps=2)
    mock_create.assert_called_once()
    assert len(prompts) == len(_PROMPTS)


def test_evaluate_scores_prompts_on_held_out_data():
    """evaluate() takes the prompts optimize() returned and scores them on another split."""
    with patch("promptolution.utils.optimize.APILLM") as mock_apillm:
        mock_apillm.return_value = _mock_llm()
        scores = evaluate(list(_PROMPTS), _DF, task_description="Classify the sentiment.")
    assert isinstance(scores, pd.DataFrame)
    assert {"prompt", "score"} <= set(scores.columns) and len(scores) == len(_PROMPTS)
    assert scores["score"].is_monotonic_decreasing  # best first


def test_optimize_then_evaluate_round_trip():
    """The two entry points compose: prompts out of optimize() go straight into evaluate()."""
    prompts = _optimize()
    with patch("promptolution.utils.optimize.APILLM") as mock_apillm:
        mock_apillm.return_value = _mock_llm()
        scores = evaluate(prompts, _DF, task_description="Classify the sentiment.")
    assert len(scores) == len(prompts)


def test_optimize_rejects_non_classification_task_type():
    with pytest.raises(NotImplementedError, match="classification"):
        optimize(_DF, task_description="Classify the sentiment.", task_type="judge")


def test_evaluate_rejects_non_classification_task_type():
    with pytest.raises(NotImplementedError, match="classification"):
        evaluate(list(_PROMPTS), _DF, task_description="Classify the sentiment.", task_type="judge")


def test_optimize_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        optimize(_DF, task_description="Classify the sentiment.", optimizer="nope")
