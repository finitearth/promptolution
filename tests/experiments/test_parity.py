"""Parity: launching via Hydra produces the SAME result as the old hand-written run_experiment call.

The experiments module is a launcher over the unchanged engine, so for the same inputs + RNG the two
paths are identical. Uses a deterministic MockLLM (no API/GPU). See examples/experiments_parity_demo.py
for a runnable, printable version.
"""

import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hydra")
from tests.mocks.mock_llm import MockLLM

from promptolution.experiments.run import compose_experiment, execute
from promptolution.helpers import run_experiment
from promptolution.utils import ExperimentConfig

_DF = pd.DataFrame(
    {
        "x": [
            "This product is amazing, I love it!",
            "Worst purchase ever, totally disappointed.",
            "It works as described, nothing special.",
            "Absolutely fantastic experience.",
            "Terrible quality, broke immediately.",
            "Pretty decent for the price.",
        ],
        "y": ["positive", "negative", "neutral", "positive", "negative", "positive"],
    }
)
_PROMPTS = [
    "Classify the sentiment of the text. Put the label between <final_answer> and </final_answer>.",
    "Is the following text positive, negative, or neutral? Answer in <final_answer> tags.",
    "Determine the sentiment of the text.",
    "Sentiment classification task:",
]


def _make_llm(*args, **kwargs):
    return MockLLM(predetermined_responses=["<final_answer>positive</final_answer>"] * 200, add_prompt_tags=True)


def _seeded():
    random.seed(0)
    np.random.seed(0)


def test_old_call_and_hydra_launch_are_identical(tmp_path):
    cfg_old = ExperimentConfig(
        optimizer="evopromptga", task_type="classification", model_id="mock", n_steps=2,
        seed=42, random_seed=42, eval_strategy="full", n_subsamples=20,
        begin_marker="<final_answer>", end_marker="</final_answer>",
        task_description="Classify the sentiment of the text as positive, negative, or neutral.",
        prompts=list(_PROMPTS), classes=["positive", "negative", "neutral"], x_column="x", y_column="y",
    )
    _seeded()
    with patch("promptolution.helpers.get_llm", side_effect=_make_llm):
        r_old = run_experiment(_DF, cfg_old).reset_index(drop=True)

    cfg_new = compose_experiment(overrides=["optimizer=evopromptga", "dataset=dummy", "n_steps=2", "random_seed=42"])
    _seeded()
    with patch("promptolution.helpers.get_llm", side_effect=_make_llm):
        out = execute(cfg_new, out_dir=tmp_path)
    r_new = pd.read_parquet(Path(out) / "prompt_scores.parquet").reset_index(drop=True)

    pd.testing.assert_frame_equal(r_old, r_new)
