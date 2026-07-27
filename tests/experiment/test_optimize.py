"""Tests for promptolution.experiment.optimize(): the lightweight, in-memory entry point."""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from tests.mocks.mock_llm import MockLLM

from promptolution.experiment import optimize

_DF = pd.DataFrame(
    {
        "x": ["I love it", "Terrible experience", "It is okay", "Absolutely amazing", "Awful and broken", "Not bad"],
        "y": ["positive", "negative", "neutral", "positive", "negative", "positive"],
    }
)
_PROMPTS = ["Classify the sentiment. <final_answer></final_answer>", "Positive, negative or neutral?"]


def _optimize(**kwargs):
    with patch("promptolution.llms.APILLM") as mock_apillm, patch(
        "promptolution.optimizers.base_optimizer.create_prompts_from_task_description"
    ) as mock_create_prompts:
        mock_apillm.return_value = MockLLM(predetermined_responses=["<final_answer>positive</final_answer>"] * 200)
        mock_create_prompts.return_value = list(_PROMPTS)
        return optimize(
            _DF,
            task_description="Classify the sentiment.",
            optimizer="evopromptga",
            n_steps=2,
            model_id="mock",
            api_key="mock",
            **kwargs,
        )


def test_optimize_in_memory_returns_frame_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _optimize()
    assert isinstance(result, pd.DataFrame) and {"prompt", "score"} <= set(result.columns)
    assert list(tmp_path.iterdir()) == []  # no output_dir -> nothing written to disk


def test_optimize_output_dir_writes_contract(tmp_path):
    out_dir = tmp_path / "run"
    result = _optimize(output_dir=out_dir)
    assert isinstance(result, pd.DataFrame) and "score" in result.columns
    for f in (".finished", "prompt_scores.parquet", "step_results.parquet", "runinfo.json"):
        assert (out_dir / f).exists(), f"missing {f}"
    info = json.loads((out_dir / "runinfo.json").read_text())
    assert info["status"] == "finished" and info["name"] == "run"  # defaulted from output_dir's name


def test_optimize_rejects_non_classification_task_type():
    with pytest.raises(NotImplementedError, match="classification"):
        optimize(_DF, task_description="Classify the sentiment.", task_type="judge")
