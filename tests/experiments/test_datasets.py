"""Dataset loader tests (in-process; pandas-only, no Hydra/HF needed)."""

import pandas as pd

from promptolution.experiments.datasets import DatasetBundle, _resolve_column, load_inline


def test_load_inline_builds_bundle():
    bundle = load_inline(
        data={"x": ["a", "b", "c"], "y": ["pos", "neg", "pos"]},
        task_description="desc",
        initial_prompts=["p1"],
        classes=["pos", "neg"],
    )
    assert isinstance(bundle, DatasetBundle)
    assert list(bundle.df.columns) == ["x", "y"]
    assert len(bundle.df) == 3
    assert bundle.task_description == "desc"
    assert bundle.initial_prompts == ["p1"]
    assert bundle.classes == ["pos", "neg"]


def test_resolve_column_by_name_and_callable():
    df = pd.DataFrame({"text": ["A", "B"], "label": ["x", "y"]})
    assert list(_resolve_column(df, "text", "x")) == ["A", "B"]
    # a callable extractor (e.g. a Hydra _partial_) computes the column
    out = _resolve_column(df, lambda d: d["text"].str.lower(), "x")
    assert list(out) == ["a", "b"]
    # empty string => no such column (e.g. reward tasks without a target)
    assert list(_resolve_column(df, "", "x").isna()) == [True, True]
