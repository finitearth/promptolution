# Components-as-interface; delete ExperimentConfig

**Status:** accepted (2026-06-22)

To run a prompt-optimization experiment, the caller constructs promptolution **components** directly
(`llm`, `task`, `predictor`, `optimizer`) — in Python for a single run, or declared in YAML and built via
Hydra `instantiate` for a grid. The old `ExperimentConfig` (a single flat config bag scattered onto
components by `apply_to`), the `get_*` string-dispatch factories, and the `config=` constructor plumbing are
**removed**. The component's constructor is the single source of truth and its schema; a wrong or misspelled
parameter fails fast at construction (deep integration approved at the 2026-06-22 review).

## Considered options
- **Keep `ExperimentConfig`** (a flat bag or a new typed config): rejected — its `apply_to` silently dropped
  callables and locally-consumed params (`VLLM.temperature`, `ClassificationTask.metric`), used one flat
  namespace for all components, and duplicated the component registry (a `Literal` + an `if/elif` dispatch
  kept in sync by hand).
- **ConfigStore / structured configs** for validation: rejected for now — a dataclass mirroring each
  constructor re-introduces exactly the duplication we're removing; `instantiate` already validates via the
  constructor.

## Consequences
- Major version bump; the public `run_experiment(df, config)` entry point and the tutorials are rewritten to
  direct construction.
- Callables (metric / reward / test_statistic) become first-class via `_target_` / `_partial_`.
