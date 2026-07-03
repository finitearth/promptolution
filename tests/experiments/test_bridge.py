"""Bridge tests: composed cfg (+ dataset bundle) -> ExperimentConfig (in-process)."""

import logging

import pandas as pd
import pytest

pytest.importorskip("hydra")
from omegaconf import OmegaConf

from promptolution.experiments.bridge import build_experiment_config
from promptolution.experiments.datasets import DatasetBundle
from promptolution.experiments.run import compose_experiment


def _bundle():
    return DatasetBundle(
        df=pd.DataFrame({"x": ["a", "b"], "y": ["positive", "negative"]}),
        task_description="desc",
        initial_prompts=["p1", "p2"],
        classes=["positive", "negative"],
    )


def test_selectors_and_scalars_mapped():
    cfg = compose_experiment(overrides=["optimizer=capo"])
    exp = build_experiment_config(cfg, _bundle())

    # selectors that drive get_* dispatch
    assert exp.optimizer == "capo"
    assert exp.task_type == "classification"
    assert exp.model_id == "mock"
    # top-level + seed (what components actually read)
    assert exp.n_steps == 3
    assert exp.seed == 42
    # scalar params gathered from groups
    assert exp.length_penalty == 0.05          # optimizer
    assert exp.begin_marker == "<final_answer>"  # predictor
    # dataset-provided experiment fields
    assert exp.task_description == "desc"
    assert exp.prompts == ["p1", "p2"]
    assert exp.classes == ["positive", "negative"]
    # plumbing keys must not leak onto the config
    assert getattr(exp, "_target_") is None
    assert getattr(exp, "name") is None


def test_reward_function_is_carried_directly():
    cfg = compose_experiment(overrides=["task=reward"])
    bundle = _bundle()
    bundle.reward_function = lambda prediction, **kw: 1.0
    exp = build_experiment_config(cfg, bundle)
    assert exp.task_type == "reward"
    assert callable(exp.reward_function)


def test_non_scalar_key_is_warned_and_dropped(caplog):
    # craft a config whose optimizer node carries a non-scalar (dict) value
    cfg = OmegaConf.create(
        {
            "llm": {"model_id": "mock"},
            "optimizer": {"name": "evopromptga", "weird": {"nested": 1}},
            "task": {"task_type": "classification"},
            "predictor": {},
            "n_steps": 2,
            "random_seed": 1,
        }
    )
    with caplog.at_level(logging.WARNING):
        exp = build_experiment_config(cfg, _bundle())
    assert getattr(exp, "weird") is None  # dropped
    assert any("not honoured under the instantiate-ready bridge" in r.message for r in caplog.records)
