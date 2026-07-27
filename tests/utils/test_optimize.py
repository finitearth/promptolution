"""Tests for promptolution.optimize(): the lightweight, in-memory entry point."""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from tests.mocks.mock_llm import MockLLM

from promptolution import optimize

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
    with patch("promptolution.llms.api_llm.APILLM") as mock_apillm:
        mock_apillm.return_value = _mock_llm()
        return optimize(
            _DF,
            task_description="Classify the sentiment.",
            optimizer="evopromptga",
            n_steps=2,
            initial_prompts=list(_PROMPTS),
            **kwargs,
        )


def test_optimize_in_memory_returns_frame_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _optimize()
    assert isinstance(result, pd.DataFrame) and {"prompt", "score"} <= set(result.columns)
    assert list(tmp_path.iterdir()) == []  # no output_dir -> nothing written to disk


def test_optimize_output_dir_writes_results(tmp_path):
    out_dir = tmp_path / "run"
    result = _optimize(output_dir=out_dir)
    assert isinstance(result, pd.DataFrame) and "score" in result.columns
    for f in ("prompt_scores.parquet", "step_results.parquet", "runinfo.json"):
        assert (out_dir / f).exists(), f"missing {f}"
    info = json.loads((out_dir / "runinfo.json").read_text())
    assert info["status"] == "finished" and info["name"] == "run"  # taken from the output dir's name


def test_optimize_generates_initial_prompts_from_task_description():
    """initial_prompts is optional; they are generated from the task description instead."""
    with patch("promptolution.optimizers.base_optimizer.create_prompts_from_task_description") as mock_create, patch(
        "promptolution.llms.api_llm.APILLM"
    ) as mock_apillm:
        mock_create.return_value = list(_PROMPTS)
        mock_apillm.return_value = _mock_llm()
        result = optimize(_DF, task_description="Classify the sentiment.", optimizer="evopromptga", n_steps=2)
    mock_create.assert_called_once()
    assert isinstance(result, pd.DataFrame) and len(result) == len(_PROMPTS)


def test_optimize_rejects_non_classification_task_type():
    with pytest.raises(NotImplementedError, match="classification"):
        optimize(_DF, task_description="Classify the sentiment.", task_type="judge")


def test_optimize_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        optimize(_DF, task_description="Classify the sentiment.", optimizer="nope")
