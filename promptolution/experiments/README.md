# `promptolution.experiments` — Hydra-based experiment gridding

Define a grid of prompt-optimization experiments once, run it locally or as a single SLURM array job,
and get per-run outputs + restart for free. This is an **optional** layer — the base library and the
end-user `run_experiment(df, config)` API stay Hydra-free.

```bash
pip install "promptolution[experiments]"   # hydra-core, hydra-submitit-launcher
```

## Quickstart

```bash
# single run (real LLM: pick llm=api or llm=vllm and provide credentials)
python -m promptolution.experiments.run optimizer=capo dataset=agnews llm=api

# offline dry run — exercises the whole pipeline with no LLM/GPU
python -m promptolution.experiments.run smoke=true

# a grid (cartesian product) — 3 x 2 x 3 = 18 runs
python -m promptolution.experiments.run -m \
    dataset=agnews,gsm8k,subj optimizer=capo,opro random_seed=42,43,44

# the same grid as ONE SLURM array job
python -m promptolution.experiments.run -m hydra/launcher=slurm \
    dataset=agnews,gsm8k optimizer=capo,opro random_seed=42,43,44
```

Programmatic / notebook use (single runs):

```python
from promptolution.experiments import compose_experiment, execute
cfg = compose_experiment(overrides=["optimizer=capo", "dataset=agnews", "smoke=true"])
run_dir = execute(cfg, out_dir="/tmp/myrun")
```

## How it's wired (instantiate-ready bridge)

`run.py:execute(cfg)` per cell: `instantiate(cfg.dataset)` → `DatasetBundle` → `bridge.build_experiment_config(cfg, bundle)` → `ExperimentConfig` → the existing `run_experiment` (with a `FileOutputCallback` writing into the Hydra run dir).

Config groups (`conf/`): `llm/`, `optimizer/`, `task/`, `dataset/`, `predictor/`. Each option file
carries a `_target_` (ready for a future switch to deep `hydra.utils.instantiate`) **plus** its params
and, where the bridge needs it, a `name`. For now the bridge flattens the composed config into the flat
`ExperimentConfig` that `run_experiment` consumes — so user-facing YAML won't change when we later flip
to deep instantiate.

### Bridge coverage (known limitation)
Scalar params and keys read directly by promptolution (`optimizer`, `task_type`, `model_id`, `n_steps`,
`seed`, `reward_function`, …) are honoured. **Callables / locally-consumed params reachable only via
`ExperimentConfig.apply_to`** — e.g. `ClassificationTask.metric`, `CAPO.test_statistic`,
`VLLM.temperature` — are not honoured under the bridge (same as today). Non-scalar config keys trigger a
warning; the temperature-style scalar case is documented in the relevant `conf/llm/*.yaml`. Datasets'
custom params/callables (`reward_function`, custom `input`/`target` extractors — e.g. `dataset=mbpp`)
**are** honoured, because they're instantiated at load time, not via `apply_to`.

## Per-run output contract

Each run directory contains:

| file | written by | purpose |
|---|---|---|
| `.hydra/config.yaml`, `overrides.yaml` | Hydra | which cell ran which config (analysis recoverability) |
| `experiment_config.yaml` | `results.py` | resolved config snapshot |
| `step_results.parquet` | `results.StepResultsCallback` | per-step trace: `step, score, prompt, input_tokens, output_tokens, time` |
| `prompt_scores.parquet` | `run.py` | final evaluated prompts + scores |
| `runinfo.json` | `results.py` | `status`, timestamps, `git_hash`, error (restart metadata) |
| `.finished` | `results.py` | completion marker → `skip_completed` reruns only unfinished cells |

Grids land under `outputs/multirun/<date>/<time>/<job_num>/` (set `PROMPTOLUTION_OUTPUT_DIR` to relocate).

## Notes for reviewers / future work
- **Logging ticket:** all run-output writing is localized in `results.py`. There is one adapter today
  (write to the run dir), so per seam discipline no multi-backend *port* is built yet — the Logging
  ticket adds the second adapter (parquet/DB/remote) and introduces the port there, in one place.
- **Analysis ticket:** the output contract above + Hydra's `.hydra/` snapshot make every result folder
  map back to its exact config; a future `load_results(sweep_dir)` reads them into one tidy DataFrame.
- **Deep instantiate:** the planned next step is to retire the bridge + `ExperimentConfig` in favour of
  `instantiate(cfg.optimizer/llm/task)` — `_target_` is already in every group config, so this is a
  local change in `run.py`/`bridge.py` with no user-facing YAML change.
