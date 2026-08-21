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

## Defining your experiment

To run on your own data, create a config directory in your project and point the CLI at it. No
dataset ships with promptolution, so this is the first step for every experiment.

```
my-paper/
  conf/task/my_tickets.yaml
  data/tickets.csv
```

A **task option** says what kind of problem you are optimizing for and where the data comes from.
Inherit the task type you want and fill in your dataset:

```yaml
# conf/task/my_tickets.yaml
defaults:
  - classification # inherits the task type and its evaluation settings
  - _self_

df:
  _target_: pandas.read_csv
  filepath_or_buffer: data/tickets.csv
x_column: review
y_column: sentiment
task_description: "Classify the review as positive or negative."
```

`df` is any `_target_` that returns a DataFrame, so loading from somewhere else means swapping the
loader. For a HuggingFace dataset:

```yaml
df:
  _target_: datasets.load_dataset
  path: SetFit/ag_news
  split: test[:300]
```

Its keys are the arguments of whatever function you named. That is why the local-file version says
`filepath_or_buffer`: it is pandas' own argument name.

Now run it, choosing a model:

```bash
promptolution-experiment --config-dir ./conf name=triage task=my_tickets llm.model_id=gpt-4o-mini
```

Your options are added to the ones that ship with promptolution rather than replacing them, so
`promptolution-experiment --config-dir ./conf --help` lists both. One consequence: a file whose name
matches a packaged option is ignored rather than merged, so give yours a name of its own.

If you would rather keep the whole run definition in one file, write your own top-level config that
inherits promptolution's and select it with `--config-name`:

```yaml
# conf/my_experiment.yaml
defaults:
  - config # promptolution's defaults
  - override task: my_tickets
  - _self_

name: triage
n_steps: 20
llm:
  model_id: gpt-4o-mini
```

```bash
promptolution-experiment --config-dir ./conf --config-name my_experiment
```

Note the `override` on the `task` line. Inside a `defaults` list it is what changes a group that the
inherited config already chose; Hydra tells you when you have left it out.

Setting a group as a plain key, outside the defaults list, is a different thing: it merges into
whichever option is selected rather than replacing it. Use it to adjust values, as `llm.model_id`
does above. To swap the option itself, use the defaults list, otherwise the old option's parameters
come along too and the error surfaces from the constructor rather than the config.

A whole grid fits in the same file, so a large sweep does not mean many files:

```yaml
# conf/my_experiment.yaml
defaults:
  - config
  - override task: my_tickets
  - _self_

name: bench
llm:
  model_id: gpt-4o-mini

hydra:
  mode: MULTIRUN # run it as a sweep without passing -m
  sweeper:
    params:
      optimizer: capo,opro
      random_seed: 42,43,44
```

## A single run

```bash
promptolution-experiment --config-dir ./conf name=my_run task=my_tickets llm.model_id=gpt-4o-mini
```

`name` is required, and it is what keys the output folder. Here is what that command does:

1. **Compose the config.** Hydra reads `conf/config.yaml`, whose `defaults` list picks one option
   per group (`llm: api`, `optimizer: capo`, `task: classification`, `predictor: marker`). Your
   `task=my_tickets` swaps the `task` group's option for the one you wrote. The result is a single
   merged config.
2. **Build the components.** Every file consists of a `_target_` plus its constructor arguments. The target is a Python path to a class or function, `hydra.utils.instantiate` calls it with the provided arguments, returning the object.
3. **Optimize and evaluate.** The task's data is split into dev and test, the optimizer runs for `n_steps` on the dev split, and the resulting
   prompts are scored on the held-out test split.
4. **Write the results.** The run's results are written into `outputs/<name>/` in the format
   described below, where `outputs/` is relative to wherever you ran the command.

To see the config a command would produce without running anything, add `--cfg job`. To list every
group and its options, run `promptolution-experiment --help`.

## Choosing a model

There is one option per LLM backend, and you pick the model with a parameter:

| Backend | Option | Needs |
| --- | --- | --- |
| Any OpenAI-compatible API | `llm=api` | `llm.model_id`, plus `llm.api_url` for non-OpenAI providers |
| vLLM, served locally | `llm=vllm` | `llm.model_id`, the `[vllm]` extra, a GPU |
| transformers, on this machine | `llm=local` | `llm.model_id`, the `[transformers]` extra |

No model is chosen for you, so `llm.model_id` is required. Only the arguments you have to supply are
listed in these files; everything else keeps the default from the Python class, and you can still set
it with a leading `+`, as in `+llm.max_tokens=1024`.

By default the same LLM proposes prompts and makes predictions. To use a different one for proposals,
fill the `meta_llm` slot from the same group:

```bash
promptolution-experiment ... llm=api llm.model_id=gpt-4o-mini \
  llm@meta_llm=api meta_llm.model_id=gpt-4o
```

`llm@meta_llm=api` means "take the `api` option from the `llm` group, but put it at `meta_llm`". Note
that plain `meta_llm=api` does something different and unhelpful: it sets `meta_llm` to the string
`"api"`, which only fails once the run starts.

## An experiment grid

Pass several values for the same key and add `-m` (short for `--multirun`). Hydra runs the cartesian
product, one cell per combination:

```bash
promptolution-experiment --config-dir ./conf -m name=bench task=my_tickets \
  llm.model_id=gpt-4o-mini optimizer=capo,opro random_seed=42,43,44
```

That is six cells (two optimizers times three seeds). Each one gets its own subdirectory under
`outputs/bench/`, named after the parameters that vary in the sweep:

```
outputs/bench/
  optimizer=capo,random_seed=42/
  optimizer=capo,random_seed=43/
  ...
```

Anything can be swept, including your datasets, since each is a task option:
`-m task=my_tickets,my_reviews`.

Models are worth a note. Within one provider the model is just a parameter, so
`-m llm.model_id=gpt-4o,gpt-4o-mini` needs no new files. Across providers the model, the URL and the
key belong together, and a sweep of independent values would pair them up wrongly, so give each
provider an option of its own:

```yaml
# conf/llm/groq_llama.yaml
_target_: promptolution.llms.APILLM
model_id: llama-3.1-8b-instant
api_url: https://api.groq.com/openai/v1
api_key: ${oc.env:GROQ_API_KEY}
```

```bash
promptolution-experiment --config-dir ./conf -m name=bench task=my_tickets llm=groq_llama,openai_mini
```

## On SLURM

The same grid becomes a single SLURM array job by switching Hydra's launcher:

```bash
promptolution-experiment --config-dir ./conf -m hydra/launcher=slurm name=bench \
  task=my_tickets llm.model_id=gpt-4o-mini optimizer=capo,opro
```

Edit `conf/hydra/launcher/slurm.yaml` for your cluster (partition, GPUs, timeout,
`array_parallelism`), or override those on the CLI: `hydra.launcher.partition=gpu`.

## Common overrides

| What                       | How                                                             |
| -------------------------- | --------------------------------------------------------------- |
| Pick a group option        | `task=my_tickets`, `optimizer=opro`, `llm=vllm`                  |
| Change a parameter         | `n_steps=20`, `llm.model_id=gpt-4o`, `task.n_subsamples=50`      |
| Add a key the config lacks | `+llm.max_tokens=1024` (the `+` is required)                     |
| Fill a second LLM slot     | `llm@meta_llm=api meta_llm.model_id=gpt-4o`                      |
| Import your own classes    | `user_dir=.`                                                     |
| Send results elsewhere     | `output_root=YOUR_OUTPUT_DIR`                                    |
| Re-run a finished cell     | `skip_completed=false`                                           |
| A value containing a comma | `task.task_description="'Classify as a, b, or c.'"`              |

That last row is not a typo. Hydra reads a bare comma as list or sweep syntax, so
`task.task_description="Classify as a, b, or c."` fails with an "Ambiguous value" error: the shell
strips the quotes and Hydra sees three items. Wrapping the value in a second, inner pair of quotes
gets literal single quotes through to Hydra, which then reads it as one string.

Prefer a config file when a combination is worth naming and reusing, and an override when it is not.
Sweeping five learning rates does not need five files.

## Credentials

Provide your API key through the environment:

```bash
export OPENAI_API_KEY=sk-...
```

Do not pass keys as plain overrides. Hydra records every override and the fully composed config inside
each run directory, so `+llm.api_key=sk-...` writes your key into `<out_dir>/.hydra/` in three places,
and those directories are what gets copied off a cluster or shared with a collaborator.

If a key has to live in a config, refer to the environment instead of the value:

```yaml
api_key: ${oc.env:GROQ_API_KEY}
```

Interpolations are saved unresolved, so the run directory records where the key came from rather than
the key itself. This is what makes an option per provider workable when each one needs its own
credentials.

## Using your own components

A `_target_` is a Python import path, so it can name your own classes as readily as promptolution's:

```python
# my-paper/mylib/optimizers.py
from promptolution.optimizers.base_optimizer import BaseOptimizer


class MyOptimizer(BaseOptimizer):
    def __init__(self, predictor, task, meta_llm=None, initial_prompts=None, callbacks=None, alpha=0.1):
        super().__init__(predictor, task, initial_prompts, callbacks)
        self.alpha = alpha

    def _step(self):
        ...
```

```yaml
# my-paper/conf/optimizer/mine.yaml
_target_: mylib.optimizers.MyOptimizer
alpha: 0.5
```

Your module has to be importable, which it is not by default: the installed command does not look in
the directory you are standing in. Point `user_dir` at the directory holding your package:

```bash
promptolution-experiment --config-dir ./conf user_dir=. name=t task=my_tickets optimizer=mine
```

If your project is installed (`pip install -e .`), the import already resolves and `user_dir` is
unnecessary. The same applies anywhere a `_target_` appears, so custom tasks, predictors and LLMs work
the same way.

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
python -m promptolution.experiment.launch --config-dir ./conf name=my_run task=my_tickets \
  llm.model_id=gpt-4o-mini
```

Run this way, the directory you are in is already importable, so custom classes work without
`user_dir`.
