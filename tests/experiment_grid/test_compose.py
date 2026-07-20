"""Config composition + fail-fast tests for the experiment_grid module."""

import pytest
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from tests.mocks.mock_llm import MockLLM
from promptolution.experiment_grid import compose_experiment


def test_defaults_compose():
    cfg = compose_experiment()
    assert cfg.llm._target_.endswith("APILLM")
    assert cfg.optimizer._target_.endswith("CAPO")
    assert cfg.task._target_.endswith("ClassificationTask")
    assert cfg.task.df._target_ == "pandas.DataFrame"  # dummy task's nested df
    assert cfg.n_steps == 10


def test_group_and_param_overrides():
    cfg = compose_experiment(overrides=["optimizer=evopromptga", "task=agnews", "n_steps=5"])
    assert cfg.optimizer._target_.endswith("EvoPromptGA")
    assert cfg.task.df.path == "SetFit/ag_news"  # agnews task's nested df loader
    assert cfg.task.df.split == "test[:300]"  # slice with all 4 classes (train[:300] is 2-class)
    assert cfg.n_steps == 5


def test_bad_param_fails_fast_at_instantiate():
    """A misspelled param blows up at construction."""
    cfg = compose_experiment(overrides=["predictor=marker", "+predictor.begin_markerr=x"])
    # instantiate wraps the underlying TypeError; assert it raises and names the bad param
    with pytest.raises(InstantiationException, match="begin_markerr"):
        instantiate(cfg.predictor, llm=MockLLM())
