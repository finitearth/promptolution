"""Tests for promptolution.utils.evaluation: train_test_split + score_prompts()."""

import pandas as pd

from tests.mocks.mock_llm import MockLLM

from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.evaluation import score_prompts, train_test_split


def _df(n=8):
    return pd.DataFrame({"x": [f"text number {i}" for i in range(n)], "y": (["positive", "negative"] * n)[:n]})


def test_train_test_split_sizes_disjoint_deterministic():
    df = _df(10)
    train, test = train_test_split(df, test_frac=0.2, seed=42)
    assert len(train) == 8 and len(test) == 2
    assert set(train["x"]) & set(test["x"]) == set()  # disjoint rows
    train2, test2 = train_test_split(df, test_frac=0.2, seed=42)
    assert test["x"].tolist() == test2["x"].tolist()  # deterministic


def test_train_test_split_normalizes_df_like():
    # A HuggingFace Dataset has .to_pandas() but no .sample(); it must be normalized first.
    class _DFLike:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

    train, test = train_test_split(_DFLike(_df(10)), test_frac=0.2, seed=42)
    assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
    assert len(train) == 8 and len(test) == 2


def test_score_prompts_scores_and_sorts():
    llm = MockLLM(predetermined_responses=["<final_answer>positive</final_answer>"] * 50)
    task = ClassificationTask(_df(), task_description="Classify the sentiment.")
    result = score_prompts(["Classify the sentiment.", "Positive or negative?"], task, MarkerBasedPredictor(llm))
    assert {"prompt", "score"} <= set(result.columns) and len(result) == 2
    assert result["score"].is_monotonic_decreasing  # best first
