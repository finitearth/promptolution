"""Parity demo: the new Hydra launcher feeds the UNCHANGED promptolution engine exactly the same
inputs as the old hand-written call — so results are identical. The Hydra layer changes only *how*
you launch an experiment, not *what* runs.

Both paths use a deterministic MockLLM (no API/GPU) and reset RNG, so the two result tables match
bit-for-bit. Run from the repo root:

    python examples/experiments_parity_demo.py
"""

import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# repo root on path so `tests.mocks` is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.mocks.mock_llm import MockLLM  # noqa: E402

from promptolution.experiments.run import compose_experiment, execute  # noqa: E402
from promptolution.helpers import run_experiment  # noqa: E402
from promptolution.utils import ExperimentConfig  # noqa: E402

# Same data + prompts as conf/dataset/dummy.yaml, so the two paths are truly comparable.
DF = pd.DataFrame(
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
PROMPTS = [
    "Classify the sentiment of the text. Put the label between <final_answer> and </final_answer>.",
    "Is the following text positive, negative, or neutral? Answer in <final_answer> tags.",
    "Determine the sentiment of the text.",
    "Sentiment classification task:",
]
RESPONSES = ["<final_answer>positive</final_answer>"] * 200


def make_llm(*args, **kwargs):
    return MockLLM(predetermined_responses=RESPONSES, add_prompt_tags=True)


def seeded():
    random.seed(0)
    np.random.seed(0)


def old_way() -> pd.DataFrame:
    """Pre-Hydra: build an ExperimentConfig by hand and call run_experiment directly."""
    cfg = ExperimentConfig(
        optimizer="evopromptga",
        task_type="classification",
        model_id="mock",
        n_steps=2,
        seed=42,
        random_seed=42,
        eval_strategy="full",
        n_subsamples=20,
        begin_marker="<final_answer>",
        end_marker="</final_answer>",
        task_description="Classify the sentiment of the text as positive, negative, or neutral.",
        prompts=list(PROMPTS),
        classes=["positive", "negative", "neutral"],
        x_column="x",
        y_column="y",
    )
    seeded()
    with patch("promptolution.helpers.get_llm", side_effect=make_llm):
        return run_experiment(DF, cfg)


def new_way() -> pd.DataFrame:
    """New: launch via Hydra (compose + execute). Same engine underneath."""
    cfg = compose_experiment(
        overrides=["optimizer=evopromptga", "dataset=dummy", "n_steps=2", "random_seed=42"]
    )
    seeded()
    with patch("promptolution.helpers.get_llm", side_effect=make_llm):
        out = execute(cfg, out_dir=Path(tempfile.mkdtemp()))
    return pd.read_parquet(out / "prompt_scores.parquet")


if __name__ == "__main__":
    r_old = old_way().reset_index(drop=True)
    r_new = new_way().reset_index(drop=True)

    print("\n=== OLD WAY (hand-written ExperimentConfig + run_experiment) ===")
    print(r_old.to_string(index=False))
    print("\n=== NEW WAY (python -m promptolution.experiments.run, via compose+execute) ===")
    print(r_new.to_string(index=False))

    identical = r_old.equals(r_new)
    print(f"\n>>> IDENTICAL RESULTS: {identical}")
    sys.exit(0 if identical else 1)
