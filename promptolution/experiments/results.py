"""Per-run output writing + restart markers (the results seam).

All run-output writing is localized here on purpose: today there is exactly one adapter (write to the
Hydra run dir), so per seam discipline we do NOT build a multi-backend port. The Logging ticket adds
the second adapter (parquet/DB/remote) and introduces the port *here*, in one place.

Per-run output contract (stable — relied on by the future Analysis & Plotting ticket):
  <run_dir>/
    .hydra/config.yaml, overrides.yaml   # written by Hydra: which cell ran which config
    step_results.parquet                 # per-step optimization trace (FileOutputCallback):
                                          #   columns: step, score, prompt, input_tokens,
                                          #            output_tokens, time
    prompt_scores.parquet                # final evaluated prompts + scores
    runinfo.json                         # status, timestamps, git_hash, tokens, error
    .finished                            # presence => run completed (skip-completed marker)
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from promptolution.utils.callbacks import BaseCallback

FINISHED_MARKER = ".finished"
RUNINFO_FILE = "runinfo.json"
STEP_RESULTS_FILE = "step_results.parquet"
_PKG_DIR = Path(__file__).resolve().parent


class StepResultsCallback(BaseCallback):
    """Write the per-step optimization trace to ``step_results.parquet`` (the output contract).

    Localized here (not the core ``FileOutputCallback``) so all run-output writing lives in one
    place — the seam the Logging ticket will extend. Engine-agnostic: rewrites the full file each
    step (avoids ``FileOutputCallback``'s fastparquet-only append, which crashes under pyarrow).
    """

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / STEP_RESULTS_FILE
        self.rows: List[dict] = []
        self.step = 0

    def on_step_end(self, optimizer) -> bool:
        self.step += 1
        llm = getattr(getattr(optimizer, "predictor", None), "llm", None)
        in_tok = getattr(llm, "input_token_count", 0)
        out_tok = getattr(llm, "output_token_count", 0)
        ts = datetime.now(timezone.utc).timestamp()
        for prompt, score in zip(optimizer.prompts, optimizer.scores):
            self.rows.append(
                {
                    "step": self.step,
                    "score": float(score),
                    "prompt": str(prompt),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "time": ts,
                }
            )
        pd.DataFrame(self.rows).to_parquet(self.path, index=False)
        return True


def is_finished(run_dir: Path) -> bool:
    """True if the run dir holds a completion marker (used for skip-completed / restart)."""
    return (Path(run_dir) / FINISHED_MARKER).exists()


def mark_finished(run_dir: Path) -> None:
    """Write the completion marker."""
    (Path(run_dir) / FINISHED_MARKER).write_text(datetime.now(timezone.utc).isoformat())


def _git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_PKG_DIR), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # pragma: no cover - git may be absent
        return None


def write_runinfo(
    run_dir: Path,
    cfg: DictConfig,
    status: str,
    error: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Write/update ``runinfo.json`` — everything needed for restart + provenance.

    Written at start (status="running") and on completion/failure so an interrupted run is
    distinguishable from a finished one.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    info_path = run_dir / RUNINFO_FILE
    if info_path.exists():
        try:
            existing = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            existing = {}

    now = datetime.now(timezone.utc).isoformat()
    info = {
        **existing,
        "status": status,
        "git_hash": _git_hash(),
        "updated_at": now,
        "started_at": existing.get("started_at", now),
        "error": error,
    }
    if status == "running" and "started_at" not in existing:
        info["started_at"] = now
    if extra:
        info.update(extra)
    info["overrides"] = list(getattr(getattr(cfg, "hydra", None), "overrides", []) or []) or info.get("overrides")
    info_path.write_text(json.dumps(info, indent=2, default=str))


def write_resolved_config(run_dir: Path, cfg: DictConfig) -> None:
    """Persist the fully-resolved experiment config (belt-and-braces alongside Hydra's .hydra/)."""
    (Path(run_dir) / "experiment_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
