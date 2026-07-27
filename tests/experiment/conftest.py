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

    def _compose(overrides=None, config_name="config"):
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            return compose(config_name=config_name, overrides=overrides or [])

    return _compose


@pytest.fixture
def cell_prompts():
    """Initial prompts injected into a cell config, to avoid the auto-generate path in tests."""
    return ["Classify the sentiment. <final_answer></final_answer>", "Positive, negative or neutral?"]


@pytest.fixture
def cfg_with_mock_llm(compose_cfg, cell_prompts, tmp_path):
    """A composed cell config wired to MockLLM and writing into ``tmp_path``."""

    def _cfg(overrides):
        cfg = compose_cfg(overrides)
        OmegaConf.set_struct(cfg, False)
        cfg.llm = {
            "_target_": "tests.mocks.mock_llm.MockLLM",
            "predetermined_responses": ["<final_answer>positive</final_answer>"] * 200,
        }
        cfg.optimizer.initial_prompts = list(cell_prompts)
        cfg.out_dir = str(tmp_path)  # replaces the interpolation; no Hydra job is running here
        return cfg

    return _cfg
