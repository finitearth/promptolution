# Running experiments

The promptolution-experiment module exists for reproducible research. It allows for config-driven runs and experiment grids, utilizing the [Hydra](https://hydra.cc) framework. Compose a prompt optimization or a whole experiment grid by defining your own config and run it through a single CLI command.

This is one of three ways to use promptolution. For a single optimization in Python, use
`promptolution.optimize()` or build the components yourself (both in the
[README](https://github.com/gepromptet/promptolution#-installation-and-quickstart)). Come here when
you want configs, sweeps, SLURM, and result files on disk.

Hydra ships with promptolution. The `experiment` extra adds the SLURM launcher plugin and
HuggingFace `datasets`:

```bash
pip install "promptolution[experiment]"
```

## Defining your experiment

To run your own experiment, create a config directory in your project next to your data:

```
my-experiment/
  conf/my_experiment.yaml
  data/tickets.csv
```

Inherit promptolution's defaults with `- promptolution`, fill in the path to your data, the task description alongside the x and y columns and the model details.

```yaml
# conf/my_experiment.yaml
defaults:
  - promptolution # the shipped defaults
  - _self_

name: triage # keys the output folder

df:
  filepath_or_buffer: data/tickets.csv

task:
  x_column: review
  y_column: sentiment
  task_description: "Classify the review as positive or negative."

llm:
  model_id: gpt-4o-mini
  api_url:
    null # null targets OpenAI; for any other provider set its base URL,
    # e.g. https://api.groq.com/openai/v1
```

Everything else comes from the shipped defaults. `promptolution-experiment --help` lists the options
each group ships. Any
constructor parameter can be set in your config even if the example omits it, so consult the API
reference for [LLMs](../api/llms.md), [Optimizers](../api/optimizers.md), [Tasks](../api/tasks.md)
and [Predictors](../api/predictors.md).

All you need to do now is provide your [credentials](#credentials) for the LLM selected in the config, and run your experiment.

## A single run

```bash
promptolution-experiment --config-dir ./conf --config-name my_experiment
```

`name` is required, and it is what keys the output folder. Here is what that command does:

1. **Compose the config.** Hydra reads your `conf/my_experiment.yaml`, which inherits
   promptolution's `promptolution.yaml` and its `defaults` list. The keys you set override what those options supply. The result is a single merged
   config.
2. **Build the components.** Every file consists of a `_target_` plus its constructor arguments. The target is a Python path to a class or function, `hydra.utils.instantiate` calls it with the provided arguments, returning the object.
3. **Optimize and evaluate.** The task's data is split into dev and test, the optimizer runs for `n_steps` on the dev split, and the resulting
   prompts are scored on the held-out test split.
4. **Write the results.** The run's results are written into `outputs/<name>/` in the format
   described below, where `outputs/` is relative to wherever you ran the command.

## An experiment grid

Pass several values for the same key and add `-m` (short for `--multirun`). Hydra runs the cartesian
product, one cell per combination:

```bash
promptolution-experiment --config-dir ./conf --config-name my_experiment -m \
  name=bench optimizer=capo,opro random_seed=42,43,44
```

That is six cells (two optimizers times three seeds). Each one gets its own subdirectory under
`outputs/bench/`, named after the overrides you passed on the command line:

```
outputs/bench/
  optimizer=capo,random_seed=42/
  optimizer=capo,random_seed=43/
  ...
```

Only the swept keys appear here because everything else lives in the config. Any extra override you
pass on the CLI lands in every cell name too, so keep a sweep's command to the keys that vary.

## On SLURM

The same grid becomes a single SLURM array job by switching Hydra's launcher:

```bash
promptolution-experiment --config-dir ./conf --config-name my_experiment -m \
  hydra/launcher=slurm name=bench optimizer=capo,opro
```

Set your cluster's parameters in your own config:

```yaml
hydra:
  launcher:
    partition: gpu
    timeout_min: 240
    array_parallelism: 16
```

or override them per command: `hydra.launcher.partition=gpu`.

## Common overrides

| What                       | How                                                                |
| -------------------------- | ------------------------------------------------------------------ |
| Pick a group option        | `optimizer=opro`, `llm=vllm_qwen2.5-7b`, `df=huggingface_datasets` |
| Change a parameter         | `n_steps=20`, `llm.model_id=gpt-4o`, `task.n_subsamples=50`        |
| Fill a second LLM slot     | `llm@meta_llm=api_gpt-4o-mini meta_llm.model_id=gpt-4o`            |
| Send results elsewhere     | `output_root=YOUR_OUTPUT_DIR`                                      |
| Re-run a finished cell     | `skip_completed=false`                                             |
| A value containing a comma | `task.task_description="'Classify as a, b, or c.'"`                |

## Credentials

Provide your API key through the environment:

```bash
export OPENAI_API_KEY=sk-...
```

This holds for any provider: with `api_key` unset, the client reads `OPENAI_API_KEY` regardless of
which `api_url` you point at. To keep provider keys in their own variables, name the variable in your
config.

Do not pass keys as plain overrides. Hydra records every override and the fully composed config inside
each run directory, so `+llm.api_key=sk-...` writes your key into `<out_dir>/.hydra/`.

Refer to the environment instead of the value:

```yaml
llm:
  api_key: ${oc.env:GROQ_API_KEY}
```

Interpolations are saved unresolved, so the run directory records where the key came from rather than
the key itself.

## Output

Every run writes the same set of files into its directory:

| File                    | Contents                                                                        |
| ----------------------- | ------------------------------------------------------------------------------- |
| `prompt_scores.parquet` | Final prompts with their held-out scores, best first                            |
| `step_results.parquet`  | Per-step trace of the optimization                                              |
| `runinfo.json`          | `name`, `status` (`running`, `finished`, `failed`), start and finish timestamps |
| `.finished`             | Restart marker, written only after a successful run                             |
| `.hydra/`               | The composed config plus the overrides that produced it                         |

A failed run still writes `runinfo.json`, with `status: failed` and the error message, and no
`.finished` marker, so a rerun will retry it.

To send results elsewhere, set `output_root`. Point the `PROMPTOLUTION_OUTPUT_DIR` environment
variable at it for a machine-wide default, or override it in a CLI command with `output_root=YOUR_OUTPUT_DIR`. The override wins if you use both.

## Restart

Output folders are keyed by `name`. Rerunning the same `name` resumes into the same folder and skips any cell that already wrote a `.finished` marker. Pass `skip_completed=false` to force re-execution.
