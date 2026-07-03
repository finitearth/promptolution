"""Dataset loaders for the experiments module.

Each ``conf/dataset/*.yaml`` points (via ``_target_``) at one of these loaders. A loader returns a
:class:`DatasetBundle` — the DataFrame plus the per-task metadata that travels with a dataset
(task description, initial prompts, classes, an optional reward function). Keeping this here makes a
grid cell *self-describing*: the YAML fully specifies how to obtain its data, including task-specific
custom params/files (e.g. MBPP's code-execution reward), handled at load time via Hydra
``instantiate`` (``_target_`` / ``_partial_``) — so it works even under the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd


@dataclass
class DatasetBundle:
    """A dataset plus the metadata that travels with it through a grid cell.

    Attributes:
        df: DataFrame with at least ``x_column`` and (for classification) ``y_column``.
        task_description: Natural-language task description (per-dataset).
        initial_prompts: Optional seed prompts; if absent the optimizer creates prompts from the
            task description (which requires a live LLM).
        classes: Optional class labels (classification).
        reward_function: Optional callable for reward tasks (``task_type: reward``).
        x_column / y_column: Column names in ``df``.
    """

    df: pd.DataFrame
    task_description: Optional[str] = None
    initial_prompts: Optional[List[str]] = None
    classes: Optional[List[str]] = None
    reward_function: Optional[Callable] = None
    x_column: str = "x"
    y_column: str = "y"
    extra: Dict[str, Any] = field(default_factory=dict)


def _resolve_column(df: pd.DataFrame, spec: Union[str, Callable], fallback: str) -> pd.Series:
    """Turn an ``input``/``target`` spec into a Series.

    ``spec`` may be a column name (str) or a callable ``(df) -> Series`` (e.g. a ``_partial_``
    extractor instantiated by Hydra). An empty string means "no such column" (e.g. reward tasks
    without a target).
    """
    if callable(spec):
        return pd.Series(spec(df)).reset_index(drop=True)
    if spec == "":
        return pd.Series([None] * len(df))
    return df[spec].reset_index(drop=True)


def load_inline(
    data: Dict[str, List[Any]],
    task_description: Optional[str] = None,
    initial_prompts: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    x_column: str = "x",
    y_column: str = "y",
) -> DatasetBundle:
    """Build a bundle from an in-memory table. Used for offline smoke runs and tests (no download)."""
    df = pd.DataFrame({k: list(v) for k, v in data.items()})
    return DatasetBundle(
        df=df,
        task_description=task_description,
        initial_prompts=list(initial_prompts) if initial_prompts else None,
        classes=list(classes) if classes else None,
        x_column=x_column,
        y_column=y_column,
    )


def load_hf(
    name: str,
    input: Union[str, Callable] = "text",
    target: Union[str, Callable] = "label",
    revision: Optional[str] = None,
    split: str = "train",
    n_samples: Optional[int] = None,
    seed: int = 42,
    task_description: Optional[str] = None,
    initial_prompts: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    reward_function: Optional[Callable] = None,
    reward_columns: Optional[List[str]] = None,
    x_column: str = "x",
    y_column: str = "y",
) -> DatasetBundle:
    """Load a HuggingFace dataset into a :class:`DatasetBundle`.

    ``input``/``target`` are either column names or callables (``_partial_`` extractors). Extra
    ``reward_columns`` are carried through verbatim for reward functions (e.g. MBPP ``test_list``).
    Requires the optional ``datasets`` package.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "load_hf requires the 'datasets' package. Install it (pip install datasets) or use "
            "an inline dataset for offline runs."
        ) from e

    ds = load_dataset(name, revision=revision, split=split)
    raw = ds.to_pandas()
    if n_samples is not None and n_samples < len(raw):
        raw = raw.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    df = pd.DataFrame()
    df[x_column] = _resolve_column(raw, input, x_column).astype(str)
    df[y_column] = _resolve_column(raw, target, y_column)
    if df[y_column].notna().any():
        df[y_column] = df[y_column].astype(str).str.lower()
    for col in reward_columns or []:
        df[col] = raw[col].reset_index(drop=True)

    return DatasetBundle(
        df=df,
        task_description=task_description,
        initial_prompts=list(initial_prompts) if initial_prompts else None,
        classes=list(classes) if classes else None,
        reward_function=reward_function,
        x_column=x_column,
        y_column=y_column,
        extra={"reward_columns": list(reward_columns or [])},
    )


# --- Example of task-specific custom params/files (Tom's MBPP case) ---------------------------------
# These are referenced from conf/dataset/mbpp.yaml via `_target_` + `_partial_` and instantiated by
# Hydra before `load_hf` is called. They demonstrate that callables/custom logic ride along with the
# dataset config — the "custom files and parameters" flexibility requested in the meeting.


def mbpp_input(df: pd.DataFrame) -> pd.Series:
    """Build the MBPP prompt input (problem statement + the function signature to implement)."""
    text = df["text"] if "text" in df.columns else df["prompt"]
    return text.astype(str)


def mbpp_reward(prediction: str, test_list: Optional[List[str]] = None, **kwargs: Any) -> float:
    """Fraction of MBPP unit tests that the generated code passes (sandbox stubbed for now).

    NOTE: executing arbitrary generated code needs a sandbox — left as a stub returning 0.0 so the
    config/plumbing is exercised without running untrusted code. The Logging/Analysis tickets and a
    real run will replace this with a sandboxed executor.
    """
    return 0.0
