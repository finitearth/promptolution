"""Tests for promptolution.utils.evaluation: dev_test_split + evaluate_prompts()."""

import pandas as pd

from tests.mocks.mock_llm import MockLLM

from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.utils.evaluation import dev_test_split, evaluate_prompts


def _df(n=8):
    return pd.DataFrame({"x": [f"text number {i}" for i in range(n)], "y": (["positive", "negative"] * n)[:n]})


def test_dev_test_split_sizes_disjoint_deterministic():
    df = _df(10)
    train, test = dev_test_split(df, test_frac=0.2, seed=42)
    assert len(train) == 8 and len(test) == 2
    assert set(train["x"]) & set(test["x"]) == set()  # disjoint rows
    dev2, test2 = dev_test_split(df, test_frac=0.2, seed=42)
    assert test["x"].tolist() == test2["x"].tolist()  # deterministic


def test_dev_test_split_normalizes_df_like():
    # A HuggingFace Dataset has .to_pandas() but no .sample(); it must be normalized first.
    class _DFLike:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

    train, test = dev_test_split(_DFLike(_df(10)), test_frac=0.2, seed=42)
    assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
    assert len(train) == 8 and len(test) == 2


def test_evaluate_prompts():
    llm = MockLLM(predetermined_responses=["<final_answer>positive</final_answer>"] * 50)
    task = ClassificationTask(_df(), task_description="Classify the sentiment.")
    result = evaluate_prompts(["Classify the sentiment.", "Positive or negative?"], task, MarkerBasedPredictor(llm))
    assert {"prompt", "score"} <= set(result.columns) and len(result) == 2
