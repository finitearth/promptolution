"""The bridge: composed Hydra config (+ dataset bundle) -> ``ExperimentConfig``.

This is the *construction seam* (DEEPENING): for this ticket it flattens the grouped Hydra config
into the flat ``ExperimentConfig`` the existing ``run_experiment`` consumes. Isolating it here means
the future flip to deep ``hydra.utils.instantiate`` is a local change — ``run.execute`` keeps calling
one ``build_*`` function, and user-facing YAML doesn't change.

Coverage (documented limitation, identical to today's behaviour): keys that promptolution reads
directly or via ``ExperimentConfig.apply_to`` *as scalars* are honoured; callables / locally-consumed
params reachable only through ``apply_to`` (``ClassificationTask.metric``, ``CAPO.test_statistic``,
``VLLM.temperature``) are not — a warning is emitted so the silent drop is visible. ``reward_function``
is honoured because ``get_task`` reads it directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from promptolution.utils import ExperimentConfig
from promptolution.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.experiments.datasets import DatasetBundle

logger = get_logger(__name__)

# Group keys that are plumbing (selection / instantiate metadata), not ExperimentConfig values.
_PLUMBING_KEYS = {"_target_", "_partial_", "_args_", "_recursive_", "_convert_", "name"}
_SCALAR_TYPES = (str, int, float, bool)


def _is_bridgeable(value: Any) -> bool:
    """Whether a value can be carried onto the flat ExperimentConfig (scalar / list / None)."""
    return value is None or isinstance(value, (*_SCALAR_TYPES, list))


def build_experiment_config(cfg: DictConfig, bundle: "DatasetBundle") -> ExperimentConfig:
    """Flatten the composed config + dataset bundle into an ``ExperimentConfig``."""
    kwargs: dict[str, Any] = {}

    # --- selectors (drive the existing get_* string dispatch) ---
    kwargs["optimizer"] = cfg.optimizer.name
    kwargs["task_type"] = cfg.task.get("task_type", "classification")
    kwargs["model_id"] = cfg.llm.model_id

    # --- top-level run params ---
    kwargs["n_steps"] = int(cfg.n_steps)
    seed = int(cfg.get("random_seed", 42))
    kwargs["seed"] = seed  # what components actually read (task.seed, vllm.seed, ...)
    kwargs["random_seed"] = seed

    # --- scalar params from each component group (apply_to / direct reads) ---
    for group in ("llm", "optimizer", "task", "predictor"):
        node = cfg.get(group)
        if node is None:
            continue
        for key, value in node.items():
            if key in _PLUMBING_KEYS or key in ("task_type", "model_id"):
                continue
            resolved = OmegaConf.to_object(value) if isinstance(value, (DictConfig, ListConfig)) else value
            if _is_bridgeable(resolved):
                kwargs.setdefault(key, resolved)
            else:
                logger.warning(
                    "🌉 bridge: config key '%s.%s' (%s) is not honoured under the instantiate-ready "
                    "bridge — it would only apply under deep instantiate. Ignored.",
                    group,
                    key,
                    type(resolved).__name__,
                )

    # --- dataset-provided experiment fields (self-describing datasets) ---
    if bundle.task_description is not None:
        kwargs["task_description"] = bundle.task_description
    if bundle.initial_prompts is not None:
        kwargs["prompts"] = list(bundle.initial_prompts)
    if bundle.classes is not None:
        kwargs["classes"] = list(bundle.classes)
    if bundle.reward_function is not None:
        kwargs["reward_function"] = bundle.reward_function  # read directly by get_task
    kwargs.setdefault("x_column", bundle.x_column)
    kwargs.setdefault("y_column", bundle.y_column)

    return ExperimentConfig(**kwargs)
