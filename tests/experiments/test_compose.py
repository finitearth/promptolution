"""Config composition + fail-fast tests for the experiments module."""

import pytest

pytest.importorskip("hydra")

from promptolution.experiments.launch import compose_experiment


def test_defaults_compose():
    cfg = compose_experiment()
    assert cfg.llm._target_.endswith("APILLM")
    assert cfg.optimizer._target_.endswith("EvoPromptGA")
    assert cfg.task._target_.endswith("ClassificationTask")
    assert cfg.task.df._target_ == "pandas.DataFrame"  # dummy task's nested df
    assert cfg.n_steps == 3


def test_group_and_param_overrides():
    cfg = compose_experiment(overrides=["optimizer=capo", "task=agnews", "n_steps=5"])
    assert cfg.optimizer._target_.endswith("CAPO")
    assert cfg.optimizer.length_penalty == 0.05
    assert cfg.task.df.path == "SetFit/ag_news"  # agnews task's nested df loader
    assert cfg.n_steps == 5


def test_bad_param_fails_fast_at_instantiate():
    """A misspelled param blows up at construction (the constructor is the schema) — not silently."""
    from hydra.utils import instantiate

    from tests.mocks.mock_llm import MockLLM

    cfg = compose_experiment(overrides=["predictor=marker", "+predictor.begin_markerr=x"])
    # instantiate wraps the underlying TypeError; assert it raises and names the bad param
    with pytest.raises(Exception, match="begin_markerr"):
        instantiate(cfg.predictor, llm=MockLLM())
