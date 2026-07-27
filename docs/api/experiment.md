# Experiment

Config-driven runs and experiment grids, via Hydra. This layer exists for reproducible research: a
run is fully described by its config, sweeping a grid is a CLI flag rather than a hand-written loop,
and a killed job resumes into the same output folder instead of starting over.

Define a run or a whole grid in `conf/` (each component is a `_target_` + its params), run it
locally or as a SLURM array job, and get per-run outputs plus the ability to restart. Hydra ships
with promptolution; the `experiments` extra adds the SLURM launcher plugin and HuggingFace
`datasets`:

```bash
pip install "promptolution[experiments]"
```

For a one-line call on a DataFrame, see `promptolution.optimize` (README quickstart) instead. For
full manual control, or for judge/reward tasks, build the components directly (README) and call
`optimizer.optimize()`.

## Quickstart

```bash
# a single run (needs an LLM: pick llm=api / llm=vllm and provide credentials); `name` is required
promptolution-experiment name=my_run task=agnews llm=api

# your own data from a CSV
promptolution-experiment name=my_run task=csv \
  task.df.filepath_or_buffer=my_data.csv task.task_description="Classify ... into: a, b."

# a grid (cartesian product) via the CLI
promptolution-experiment -m name=bench task=agnews optimizer=capo,opro random_seed=42,43,44

# the same grid as ONE SLURM array job
promptolution-experiment -m hydra/launcher=slurm name=bench task=agnews optimizer=capo,opro
```

No install, or want to run straight from a checkout? The same CLI also runs as a module:
`python -m promptolution.experiment.launch name=my_run task=agnews llm=api`.

## How it works

`execute(cfg)` builds the components with `hydra.utils.instantiate`:
`llm → predictor(llm) → task(df) → optimizer(predictor, meta_llm, task)`, splits the task's data into
train/test, optimizes, evaluates the final prompts on the held-out split, and writes the output
contract (below) to `cfg.out_dir`, the run directory Hydra creates for this cell.

**Config groups** (`conf/`): one `_target_` + params per option.

```
conf/
  config.yaml            defaults + name + n_steps + test_frac + restart/output settings
  llm/        api · vllm
  optimizer/  capo · opro · evopromptga · evopromptde
  task/       demo · agnews · csv       # the Task carries its data as a nested `df:` _target_
  predictor/  marker · first_occurrence
  hydra/launcher/ slurm             # -m hydra/launcher=slurm  -> one SLURM array job
```

### Data lives on the Task

A Task's `df` is a nested `_target_` returning a DataFrame: pandas for files/inline, or
`datasets.load_dataset` for HuggingFace.

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

`name` is **required** and keys the output folder: `<PROMPTOLUTION_OUTPUT_DIR|outputs>/<name>/` (no
timestamp). Each grid cell's subdir is the slug of its swept params. Rerunning the same `name`
**resumes into the same folder**, skipping cells that already wrote a `.finished` marker
(`skip_completed: true`).

## Reference

::: promptolution.experiment.launch
    options:
      show_root_heading: true
