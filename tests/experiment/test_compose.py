"""Config composition + fail-fast tests for the experiment module."""

import pytest
from hydra.errors import InstantiationException
from hydra.utils import instantiate

from tests.mocks.mock_llm import MockLLM


def test_defaults_compose(compose_cfg):
    cfg = compose_cfg()
    assert cfg.llm._target_.endswith("APILLM")
    assert cfg.optimizer._target_.endswith("CAPO")
    assert cfg.task._target_.endswith("ClassificationTask")
    assert cfg.task.df._target_ == "pandas.DataFrame"  # demo dataset's nested df
    assert cfg.n_steps == 10


def test_group_and_param_overrides(compose_cfg):
    cfg = compose_cfg(overrides=["optimizer=evopromptga", "data=agnews", "n_steps=5"])
    assert cfg.optimizer._target_.endswith("EvoPromptGA")
    assert cfg.task.df.path == "SetFit/ag_news"  # agnews dataset's nested df loader
    assert cfg.task.df.split == "test[:300]"  # slice with all 4 classes (train[:300] is 2-class)
    assert cfg.n_steps == 5


def test_data_group_feeds_the_task(compose_cfg):
    """The task type is one group, the dataset another; the task pulls the data in by interpolation."""
    cfg = compose_cfg(overrides=["data=agnews"])
    assert cfg.task._target_.endswith("ClassificationTask")  # task group holds only the task type
    assert cfg.task.x_column == cfg.data.x_column == "text"  # dataset columns come from the data group
    assert cfg.task.task_description == cfg.data.task_description


def test_data_csv_loads_own_data(compose_cfg, tmp_path):
    """data=csv reads a file with pandas and feeds it to the task."""
    csv = tmp_path / "mine.csv"
    csv.write_text("x,y\ngreat,positive\nawful,negative\n")
    cfg = compose_cfg(overrides=["data=csv", f"data.df.filepath_or_buffer={csv}", "data.task_description=Classify."])
    df = instantiate(cfg.task.df)
    assert list(df.columns) == ["x", "y"] and len(df) == 2
    assert cfg.task.task_description == "Classify."


def test_bad_param_fails_fast_at_instantiate(compose_cfg):
    """A misspelled param blows up at construction."""
    cfg = compose_cfg(overrides=["predictor=marker", "+predictor.begin_markerr=x"])
    # instantiate wraps the underlying TypeError; assert it raises and names the bad param
    with pytest.raises(InstantiationException, match="begin_markerr"):
        instantiate(cfg.predictor, llm=MockLLM())
