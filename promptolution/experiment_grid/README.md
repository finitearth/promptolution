# `promptolution.experiments` — config-driven runs & grids (Hydra)

The **config/CLI layer** over the Hydra-free core runner (`promptolution.runner.run`). Define a run — or a
whole grid — in `conf/` (each component is a `_target_` + its params), run it locally or as a SLURM array
job, and get per-run outputs + restart for free. Optional layer:

```bash
pip install "promptolution[experiments]"   # hydra-core, hydra-submitit-launcher, datasets
```

A plain single run needs **no** Hydra — just build the components and call `promptolution.runner.run`
(see the top-level README). This package is for config-driven runs and grids.

## Quickstart

```bash
# a single run (needs an LLM: pick llm=api / llm=vllm and provide credentials); `name` is required
python -m promptolution.experiments.launch name=my_run optimizer=capo task=agnews llm=api

# a grid (cartesian product) via the CLI
python -m promptolution.experiments.launch -m name=bench optimizer=capo,opro random_seed=42,43,44

# the same grid as ONE SLURM array job
python -m promptolution.experiments.launch -m hydra/launcher=slurm name=bench optimizer=capo,opro
```

Programmatic (notebooks):

```python
from promptolution.experiments import run_grid, execute, compose_experiment

# a small local grid (serial — for parallelism/SLURM use the CLI -m)
results = run_grid({"optimizer": ["capo", "opro"], "random_seed": [42, 43]},
                   overrides=["task=agnews", "llm=api"], name="bench")
```

## How it works

`launch.execute(cfg)` builds the components with `hydra.utils.instantiate`:
`llm → predictor(llm) → task(df) → optimizer(predictor, meta_llm, task)`, splits the task's data into
train/test, and delegates to `runner.run`.

**Config groups** (`conf/`): one `_target_` + params per option.

```
conf/
  config.yaml            defaults + name + n_steps + test_frac + restart/output settings
  llm/        api · vllm
  optimizer/  capo · opro · evopromptga · evopromptde
  task/       dummy · agnews        # the Task carries its data as a nested `df:` _target_
  predictor/  marker
  hydra/launcher/ slurm             # -m hydra/launcher=slurm  -> one SLURM array job
  grid_example.yaml                 # a whole grid defined in a file (hydra.sweeper.params)
```

### Data lives on the Task

A Task's `df` is a nested `_target_` returning a DataFrame — pandas for files/inline, or
`datasets.load_dataset` for HuggingFace (a `Dataset` is normalized via `.to_pandas()` in `BaseTask`):

```yaml
# conf/task/mydata.yaml
_target_: promptolution.tasks.ClassificationTask
df: { _target_: pandas.read_parquet, filepath_or_buffer: data/mine.parquet } # or datasets.load_dataset / your own loader
x_column: text
y_column: label
task_description: "..."
```

Override just the data from the CLI: `task.df.filepath_or_buffer=other.parquet`.

## Restart

`name` is **required** and keys the output folder — `<PROMPTOLUTION_OUTPUT_DIR|outputs>/<name>/` (no
timestamp). Each grid cell's subdir is the slug of its swept params. Rerunning the same `name` **resumes
into the same folder**, skipping cells that already wrote a `.finished` marker (`skip_completed: true`).

## Per-run output contract

Each run dir holds `.hydra/config.yaml` + `overrides.yaml` (Hydra, CLI path only), `step_results.parquet`
(per-step trace), `prompt_scores.parquet` (final evaluated prompts), `runinfo.json` (status/timestamps),
and `.finished`.
