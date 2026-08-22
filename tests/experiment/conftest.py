"""Fixtures for the experiment tests: composing the shipped conf/ and a mock-LLM cell config."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import promptolution.experiment

CONFIG_DIR = str(Path(promptolution.experiment.__file__).resolve().parent / "conf")


@pytest.fixture
def compose_cfg():
    """Compose the shipped conf/ as the CLI would, but without running a Hydra job.

    ``cfg.out_dir`` is a ``${hydra:...}`` interpolation that only resolves inside a job, so any test
    that runs a cell has to assign over it -- see `cfg_with_mock_llm`.
    """

    def _compose(overrides=None, config_name="promptolution"):
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            return compose(config_name=config_name, overrides=overrides or [])

    return _compose


@pytest.fixture
def cell_prompts():
    """Initial prompts injected into a cell config, to avoid the auto-generate path in tests."""
    return ["Classify the sentiment. <final_answer></final_answer>", "Positive, negative or neutral?"]


@pytest.fixture
def cell_dataset():
    """An inline dataset node, standing in for the option a user would define in their own conf/."""
    return {
        "_target_": "pandas.DataFrame",
        "_convert_": "all",  # hand pandas plain dict/lists rather than OmegaConf containers
        "data": {
            "x": [
                "I love it",
                "Terrible experience",
                "It is okay",
                "Absolutely amazing",
                "Awful and broken",
                "Not bad at all",
                "Great stuff",
                "Poor quality",
            ],
            "y": ["positive", "negative", "neutral", "positive", "negative", "positive", "positive", "negative"],
        },
    }


@pytest.fixture
def cfg_with_mock_llm(compose_cfg, cell_prompts, cell_dataset, tmp_path):
    """A composed cell config wired to MockLLM and an inline dataset, writing into ``tmp_path``.

    No dataset ships with the package: the ``df`` group needs a path and the task needs its columns
    and description, so the fixture supplies them the same way it supplies the LLM.
    """

    def _cfg(overrides=None):
        cfg = compose_cfg(overrides)
        OmegaConf.set_struct(cfg, False)
        cfg.llm = {
            "_target_": "tests.mocks.mock_llm.MockLLM",
            "predetermined_responses": ["<final_answer>positive</final_answer>"] * 200,
        }
        cfg.df = cell_dataset  # the dataset is its own top-level group
        cfg.task.x_column = "x"
        cfg.task.y_column = "y"
        cfg.task.task_description = "Classify the sentiment of the text as positive, negative, or neutral."
        cfg.optimizer.initial_prompts = list(cell_prompts)
        cfg.out_dir = str(tmp_path)  # replaces the interpolation; no Hydra job is running here
        return cfg

    return _cfg
