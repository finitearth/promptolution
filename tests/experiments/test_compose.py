"""Config composition tests (in-process — config building is pure)."""

import pytest

hydra = pytest.importorskip("hydra")

from promptolution.experiments.run import compose_experiment


def test_defaults_compose():
    cfg = compose_experiment()
    assert cfg.optimizer.name == "evopromptga"
    assert cfg.task.task_type == "classification"
    assert cfg.dataset._target_.endswith("load_inline")
    assert cfg.n_steps == 3
    assert cfg.smoke is False


def test_group_override_swaps_option():
    cfg = compose_experiment(overrides=["optimizer=capo", "llm=vllm"])
    assert cfg.optimizer.name == "capo"
    assert cfg.optimizer.length_penalty == 0.05  # capo-specific param present
    assert cfg.llm.model_id.startswith("vllm-")


def test_param_override():
    cfg = compose_experiment(overrides=["n_steps=7", "random_seed=99"])
    assert cfg.n_steps == 7
    assert cfg.random_seed == 99
