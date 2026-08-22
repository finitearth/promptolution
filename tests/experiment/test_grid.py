"""Restart behavior of launch(): name-keyed dir, skip finished cells."""

from promptolution.experiment.launch import launch


def test_restart_skips_finished(cfg_with_mock_llm, tmp_path):
    """A second run into the same folder skips the finished cell (does not re-execute)."""
    cfg = cfg_with_mock_llm(["name=t", "optimizer=evopromptga", "n_steps=2", "test_frac=0.25"])
    launch(cfg)
    assert (tmp_path / ".finished").exists()
    (tmp_path / "step_results.parquet").unlink()  # remove an output
    launch(cfg)  # skip_completed default -> skip
    assert not (tmp_path / "step_results.parquet").exists()  # not regenerated => skipped
