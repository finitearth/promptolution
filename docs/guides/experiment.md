# Running experiments

Config-driven runs and experiment grids, via [Hydra](https://hydra.cc). This module exists for
reproducible research. Define a run or a whole experiment grid through nested config files, then launch it with a single CLI command.

This is one of three ways to use promptolution. For a single optimization in Python, use
`promptolution.optimize()` or build the components yourself (both in the
[README](https://github.com/gepromptet/promptolution#-installation-and-quickstart)). Come here when
you want configs, sweeps, SLURM, and result files on disk.

Hydra ships with promptolution. The `experiment` extra adds the SLURM launcher plugin and
HuggingFace `datasets`:

```bash
pip install "promptolution[experiment]"
```

## A single run

```bash
promptolution-experiment name=my_run data=agnews
```

`name` is required, and it is what keys the output folder. Here is what that command does:

1. **Compose the config.** Hydra reads `conf/config.yaml`, whose `defaults` list picks one option
   per group (`llm: api`, `optimizer: capo`, `data: demo`, `task: classification`,
   `predictor: marker`). Your `data=agnews` swaps the `data` group's option, so
   `conf/data/agnews.yaml` is used instead of `demo.yaml`. The result is a single merged config.
2. **Build the components.** Every file consists of a `_target_` plus its constructor arguments. The target is a Python path to a class or function, `hydra.utils.instantiate` calls it with the provided arguments, returning the object.
3. **Optimize and evaluate.** The task's data is split into train and test, the optimizer runs for `n_steps` on the train split, and the resulting
   prompts are scored on the held-out split.
4. **Write the results.** The run's results are written into `outputs/<name>/` in the format
   described below, where `outputs/` is relative to wherever you ran the command.

To see the config a command would produce without running anything, add `--cfg job`. To list every
group and its options, run `promptolution-experiment --help`.

## An experiment grid

Pass several values for the same key and add `-m` (short for `--multirun`). Hydra runs the cartesian
product, one cell per combination:

```bash
promptolution-experiment -m name=bench data=agnews optimizer=capo,opro random_seed=42,43,44
```

That is six cells (two optimizers times three seeds). Each one gets its own subdirectory under
`outputs/bench/`, named after the parameters that vary in the sweep:

```
outputs/bench/
  data=agnews,optimizer=capo,random_seed=42/
  data=agnews,optimizer=capo,random_seed=43/
  ...
```

## On SLURM

The same grid becomes a single SLURM array job by switching Hydra's launcher:

```bash
promptolution-experiment -m hydra/launcher=slurm name=bench data=agnews optimizer=capo,opro
```

Edit `conf/hydra/launcher/slurm.yaml` for your cluster (partition, GPUs, timeout,
`array_parallelism`), or override those on the CLI: `hydra.launcher.partition=gpu`.

## Common overrides

| What                       | How                                                            |
| -------------------------- | -------------------------------------------------------------- |
| Pick a group option        | `data=agnews`, `optimizer=opro`, `llm=vllm`                    |
| Change a parameter         | `n_steps=20`, `llm.model_id=gpt-4o`, `optimizer.upper_shots=3` |
| Add a key the config lacks | `+optimizer.alpha=0.1` (the `+` is required)                   |
| Send results elsewhere     | `output_root=YOUR_OUTPUT_DIR`                                  |
| Re-run a finished cell     | `skip_completed=false`                                         |
| A value containing a comma | `data.task_description="'Classify as a, b, or c.'"`            |

That last row is not a typo. Hydra reads a bare comma as list or sweep syntax, so
`data.task_description="Classify as a, b, or c."` fails with an "Ambiguous value" error: the shell
strips the quotes and Hydra sees three items. Wrapping the value in a second, inner pair of quotes
gets literal single quotes through to Hydra, which then reads it as one string.

## Credentials

Provide your API key through the environment:

```bash
export OPENAI_API_KEY=sk-...
```

Do not pass keys as overrides as Hydra records every override and the fully composed config inside each run directory. Use the environment variable instead.

For a non-OpenAI provider, set the base URL on the LLM config: `llm.api_url=https://...`.

## Bringing your own data

The `data` group carries the dataset: how to load it, which columns hold the input and the label,
and how the task is described to the meta-LLM. `data=csv` reads a local file or URL with pandas:

```bash
promptolution-experiment name=my_run data=csv \
  data.df.filepath_or_buffer=my_data.csv \
  data.task_description="'Classify the ticket as account, billing, or bug.'" \
  data.x_column=review data.y_column=sentiment
```

`df` is a nested `_target_`, so its keys are the arguments of whatever loads the data. That is why
the path is `filepath_or_buffer`: it is pandas' own argument name for it.

For another format or source, add your own option to the group. A `data` option needs a `df` consisting of a loader function specified in `_target_`, and the arguments this function takes, plus the three metadata fields:

```yaml
# conf/data/mydata.yaml
df:
  _target_: datasets.load_dataset # or pandas.read_parquet, pandas.read_sql, your own function
  path: my-org/my-dataset
  additional_arg: 42
x_column: text
y_column: label
task_description: "..."
```

Then use it with `data=mydata`.

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

This matters most for grids on a cluster: if a job hits its wall time halfway through a sweep,
resubmitting the identical command picks up only the cells that did not finish.

## Running without installing

The console script comes from the installed package. From a checkout, the same CLI runs as a module:

```bash
python -m promptolution.experiment.launch name=my_run data=agnews
```
