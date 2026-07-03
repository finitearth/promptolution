# promptolution

Prompt-optimization library and a zoo of optimizers. This glossary covers the language of running
experiments — a single prompt optimization or a whole grid of them.

## Language

**Component**:
An `llm`, `task`, `predictor`, or `optimizer` — a promptolution object built directly in Python or via
Hydra `instantiate`. Components *are* the interface: single runs and grid cells are both assembled from them.
_Avoid_: config, module, unit.

**Runner**:
The one function that executes a single experiment — optimize the optimizer's train **Task**, evaluate the
best prompts on an optional `test_df`, write the output contract, and handle restart. Shared by single runs
and every grid **Cell**.
_Avoid_: engine, executor, pipeline.

**Cell**:
One point of a **Grid** — a single fully-specified set of component overrides, i.e. exactly one run.
_Avoid_: job, node, task (a "task" is a Component).

**Grid**:
The set of **Cells** formed by sweeping component parameters (Hydra multirun, a YAML sweep, or `run_grid`).
_Avoid_: sweep (use for the act), batch.

**Experiment**:
A **named** run or grid. Its name keys the output folder, which is what makes restart resume into the *same*
folder (finished **Cells** skipped).
_Avoid_: run id, job name.

**Direct construction**:
The single-run style: build **Components** in Python and call `optimizer.optimize()` — no config object.
_Avoid_: quickstart config, `run_experiment(df, config)` (retired).

## Flagged ambiguities / retired terms

- **ExperimentConfig** — *retired* (deleted). It was a single flat config bag scattered onto components via
  `apply_to`. Do not reintroduce a global config object; a Component's constructor is its schema.
- **Bridge** — *retired* (deleted). Was the flat→nested mapping from a composed config onto `ExperimentConfig`.
  With deep `instantiate` there is no such mapping.

## Example dialogue

> **Dev:** "So to run one optimization I make an `ExperimentConfig` and call `run_experiment`?"
> **Maintainer:** "No — that's retired. You build the **Components**: an `llm`, a `task` from your `train_df`,
> an `optimizer`, then `optimizer.optimize()`. That's **direct construction**."
> **Dev:** "And a **Grid**?"
> **Maintainer:** "Same Components, but declared in YAML and built by `instantiate`. Sweeping them gives
> **Cells**; the shared **Runner** executes each one. Name the **Experiment** and a rerun resumes into the
> same folder, skipping finished Cells."
> **Dev:** "Where does the dataset go?"
> **Maintainer:** "It's the **Task**'s `df`. In Python you pass it; in a grid it's a nested `_target_` on the
> Task config — there's no separate dataset object."
