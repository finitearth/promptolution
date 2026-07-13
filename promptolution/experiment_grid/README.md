# `promptolution.experiment_grid` — config-driven runs & grids (Hydra)

Define a run — or a whole grid — in `conf/` (each component is a `_target_` + its params), run it
locally or as a SLURM array job, and get per-run outputs + restart for free. Hydra ships with
promptolution; the extra adds the SLURM launcher plugin and HuggingFace `datasets`:

```bash
pip install "promptolution[experiments]"   # hydra-submitit-launcher, datasets
```

This layer is for reproducible research experiments. For a plain "best prompt for my task" run in
Python, use the one-call `promptolution.optimize.optimize` (see the top-level README).

## Quickstart

```bash
# a single run (needs an LLM: pick llm=api / llm=vllm and provide credentials); `name` is required
python -m promptolution.experiment_grid name=my_run task=agnews llm=api

# your own data from a CSV
python -m promptolution.experiment_grid name=my_run task=csv \
  task.df.filepath_or_buffer=my_data.csv task.task_description="Classify ... into: a, b."

# a grid (cartesian product) via the CLI
python -m promptolution.experiment_grid -m name=bench optimizer=capo,opro random_seed=42,43,44

# the same grid as ONE SLURM array job
python -m promptolution.experiment_grid -m hydra/launcher=slurm name=bench optimizer=capo,opro
```

Programmatic (notebooks/tests): `compose_experiment(overrides)` builds a config, `execute(cfg, out_dir)`
runs one cell. Grids run through the Hydra CLI (`-m`).

## How it works

`execute(cfg, out_dir)` builds the components with `hydra.utils.instantiate`:
`llm → predictor(llm) → task(df) → optimizer(predictor, meta_llm, task)`, splits the task's data into
train/test, optimizes, evaluates the final prompts on the held-out split, and writes the output
contract (below) to `out_dir`.

**Config groups** (`conf/`): one `_target_` + params per option.

```
conf/
  config.yaml            defaults + name + n_steps + test_frac + restart/output settings
  llm/        api · vllm
  optimizer/  capo · opro · evopromptga · evopromptde
  task/       dummy · agnews · csv      # the Task carries its data as a nested `df:` _target_
  predictor/  marker · first_occurrence
  hydra/launcher/ slurm             # -m hydra/launcher=slurm  -> one SLURM array job
  grid_example.yaml                 # a whole grid defined in a file (hydra.sweeper.params)
```

### Data lives on the Task

A Task's `df` is a nested `_target_` returning a DataFrame — pandas for files/inline, or
`datasets.load_dataset` for HuggingFace (a `Dataset` is normalized via `.to_pandas()` in
`train_test_split`):

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

`execute` writes to each run dir: `step_results.parquet` (per-step trace), `prompt_scores.parquet`
(final evaluated prompts), `runinfo.json` (name/status/timestamps), and `.finished`. The CLI path adds
Hydra's `.hydra/config.yaml` + `overrides.yaml` snapshot.
