"""Restart behavior of execute(): name-keyed dir, skip finished cells."""

from omegaconf import OmegaConf

from promptolution.experiment_grid import compose_experiment, execute

_PROMPTS = ["Classify the sentiment. <final_answer></final_answer>", "Positive, negative or neutral?"]


def _cfg_with_mock_llm(overrides):
    cfg = compose_experiment(overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.llm = {
        "_target_": "tests.mocks.mock_llm.MockLLM",
        "predetermined_responses": ["<final_answer>positive</final_answer>"] * 200,
    }
    cfg.optimizer.initial_prompts = list(_PROMPTS)
    return cfg


def test_restart_skips_finished(tmp_path):
    """A second run into the same folder skips the finished cell (does not re-execute)."""
    cfg = _cfg_with_mock_llm(["name=t", "task=dummy", "optimizer=evopromptga", "n_steps=2", "test_frac=0.25"])
    execute(cfg, out_dir=tmp_path)
    assert (tmp_path / ".finished").exists()
    (tmp_path / "step_results.parquet").unlink()  # remove an output
    execute(cfg, out_dir=tmp_path)  # skip_completed default -> skip
    assert not (tmp_path / "step_results.parquet").exists()  # not regenerated => skipped
