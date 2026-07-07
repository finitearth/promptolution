"""Tests for the shared runner (promptolution.runner): train_test_split + run()."""

import json

import pandas as pd

from tests.mocks.mock_llm import MockLLM

from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor
from promptolution.runner import run, train_test_split
from promptolution.tasks.classification_tasks import ClassificationTask


def _df(n=8):
    return pd.DataFrame(
        {"x": [f"text number {i}" for i in range(n)], "y": (["positive", "negative"] * n)[:n]}
    )


def test_train_test_split_sizes_disjoint_deterministic():
    df = _df(10)
    train, test = train_test_split(df, test_frac=0.2, seed=42)
    assert len(train) == 8 and len(test) == 2
    assert set(train["x"]) & set(test["x"]) == set()          # disjoint rows
    train2, test2 = train_test_split(df, test_frac=0.2, seed=42)
    assert test["x"].tolist() == test2["x"].tolist()          # deterministic


def _optimizer(df):
    llm = MockLLM(predetermined_responses=["<final_answer>positive</final_answer>"] * 200)
    task = ClassificationTask(df, task_description="Classify the sentiment.")
    return EvoPromptGA(
        predictor=MarkerBasedPredictor(llm),
        meta_llm=llm,
        task=task,
        initial_prompts=["Classify the sentiment.", "Positive or negative?"],
    )


def test_run_writes_output_contract(tmp_path):
    result = run(_optimizer(_df()), n_steps=2, output_dir=tmp_path, name="t")
    assert isinstance(result, pd.DataFrame) and {"prompt", "score"} <= set(result.columns)
    for f in (".finished", "prompt_scores.parquet", "step_results.parquet", "runinfo.json"):
        assert (tmp_path / f).exists(), f"missing {f}"
    info = json.loads((tmp_path / "runinfo.json").read_text())
    assert info["status"] == "finished" and info["name"] == "t"


def test_run_skips_completed(tmp_path):
    run(_optimizer(_df()), n_steps=2, output_dir=tmp_path, name="t")
    (tmp_path / "step_results.parquet").unlink()              # remove an output
    run(_optimizer(_df()), n_steps=2, output_dir=tmp_path, name="t")  # .finished present -> skip
    assert not (tmp_path / "step_results.parquet").exists()   # not regenerated => skipped


def test_run_evaluates_on_test_task(tmp_path):
    df = _df(12)
    train_df, test_df = train_test_split(df, test_frac=0.25)
    llm = MockLLM(predetermined_responses=["<final_answer>positive</final_answer>"] * 200)
    train_task = ClassificationTask(train_df, task_description="Classify the sentiment.")
    test_task = ClassificationTask(test_df, task_description=train_task.task_description)
    opt = EvoPromptGA(
        predictor=MarkerBasedPredictor(llm), meta_llm=llm, task=train_task,
        initial_prompts=["Classify the sentiment.", "Positive or negative?"],
    )
    result = run(opt, n_steps=2, test_task=test_task, output_dir=tmp_path, name="t")
    assert len(result) == 2  # both prompts scored on the held-out test task
