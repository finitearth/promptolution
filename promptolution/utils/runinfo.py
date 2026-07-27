"""Persist a run's status and timestamps to runinfo.json (part of the experiment output contract)."""

import json
from datetime import datetime, timezone
from pathlib import Path

from typing import Any, Dict, Union


def start_runinfo(out_dir: Union[str, Path], name: str) -> Dict[str, Any]:
    """Write a "running" runinfo.json with the current UTC timestamp.

    Args:
        out_dir (Union[str, Path]): Directory runinfo.json is written to.
        name (str): The run's name.

    Returns:
        Dict[str, Any]: The info dict, to be passed to :func:`finish_runinfo` later.
    """
    info = {"name": name, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    _write(out_dir, info)
    return info


def finish_runinfo(out_dir: Union[str, Path], info: Dict[str, Any], status: str, **extra: Any) -> Dict[str, Any]:
    """Update info with a finished/failed status and the current UTC timestamp, and rewrite runinfo.json.

    Args:
        out_dir (Union[str, Path]): Directory runinfo.json is written to.
        info (Dict[str, Any]): The dict returned by :func:`start_runinfo`.
        status (str): The run's final status, e.g. "finished" or "failed".
        **extra (Any): Extra fields to record, e.g. ``error=str(e)``.

    Returns:
        Dict[str, Any]: The updated info dict.
    """
    info.update(status=status, finished_at=datetime.now(timezone.utc).isoformat(), **extra)
    _write(out_dir, info)
    return info


def _write(out_dir: Union[str, Path], info: Dict[str, Any]) -> None:
    (Path(out_dir) / "runinfo.json").write_text(json.dumps(info, indent=2))
