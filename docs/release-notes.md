# Release Notes

## Release v1.4.0
### What's changed
#### Added features
* Reworked APILLM to allow for calls to any API that follows the OpenAI API format
* Added graceful failing in optimization runs, allowing to obtain results after an error
* Reworked configs to ExperimentConfig, allowing to parse any attributes

### Further Changes:
* Reworked getting started notebook
* Added tests for the entire package, covering roughly 80% of the codebase
* Reworked dependency and import structure to allow the usage of a subset of the package

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v1.3.2...v1.4.0)

## Release v1.3.2
### What's changed
#### Added features
* Allow for configuration and evaluation of system prompts in all LLM-Classes
* CSV Callback is now FileOutputCallback and able to write Parquet files
* Fixed LLM-Call templates in VLLM
* refined OPRO-implementation to be closer to the paper

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v1.3.1...v1.3.2)

## Release v1.3.1
### What's changed
#### Added features
* new features for the VLLM Wrapper (accept seeding to ensure reproducibility)
* fixes in the "MarkerBasedClassificator"
* fixes in prompt creation and task description handling
* generalize the Classificator
* add verbosity and callback handling in EvoPromptGA
* add timestamp to the callback
* removed datasets from repo
* changed task creation (now by default with a dataset)

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v1.3.0...v1.3.1)

## Release v1.3.0
### What's changed
#### Added features
* new features for the VLLM Wrapper (automatic batch size determination, accepting kwargs)
* allow callbacks to terminate optimization run
* add token count functionality
* renamed "Classificator"-Predictor to "FirstOccurenceClassificator"
* introduced "MarkerBasedClassifcator"
* automatic task description creation
* use task description in prompt creation
* implement CSV callbacks

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v1.2.0...v1.3.0)

## Release v1.2.0
### What's changed
#### Added features
* New LLM wrapper: VLLM for local inference with batches

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v1.1.1...v1.2.0)

## Release v1.1.1
### What's Changed
#### Further Changes:
- deleted poetry.lock
- updated transformers dependency: bumped from 4.46.3 to 4.48.0

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v1.1.0...v1.1.1)

## Release v1.1.0
### What's changed
#### Added features
* Enable reading tasks from a pandas dataframe

#### Further Changes:
* deleted experiment files from the repo folders (logs, configs, etc.)
* improved opros meta-prompt
* added support for python versions from 3.9 onwards (previously 3.11)

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v1.0.1...v1.1.0)

## Release v1.0.1
### What's changed
#### Added features
-

#### Further Changes:
* fixed release notes

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v1.0.0...v1.0.1)

## Release v1.0.0
### What's changed
#### Added Features:
* exemplar selection module, classes for exemplar selection (Random and RandomSearch)
* helper functions: run_experiment, run_optimization and run_evaluation

#### Further Changes:
* removed deepinfra helper functions as langchain-community libary is now working as intended
* added license
* added release notes :)

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v0.2.0...v1.0.0)

## Release v0.2.0

### What's Changed
#### Added Features: 
* Prompt creation utility function
* Prompt variation utility function
* New optimizer: OPro (see [arXiv paper](https://arxiv.org/abs/2309.03409))

#### Further Changes:
* Workflows for automated build, deployment & release
* New documentation page appearance
* Additional Docstrings & Formatting

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/v0.1.1...v0.2.0)

## Release v0.1.1 (2)

### What's Changed

#### Added features:
\-

#### Further changes:
* Added workflows for automated build, deployment, release and doc creation
* Updated pre-commits
* Added docstrings and formatting
* Updated readme
* Updated docs

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/0.1.1...v0.1.1)

## Release v0.1.1

### What's Changed

#### Features added:
\-

#### Further changes:
* Loosen restrictive python version requirements (^3.11 instead of ~3.11)
* Add documentation pages
* Update README

**Full Changelog**: [here](https://github.com/finitearth/promptolution/compare/0.1.0...0.1.1)

## Release v0.1.0

*First release*

### What's Changed

#### Added Features:
* Base classes for tasks, LLMs, predictors, and optimizers
* Classification task
* API LLMs from OpenAI, Anthropic, and DeepInfra
* Local LLM
* optimizer EvoPrompt GA and EvoPrompt DE (see [arXiv paper](https://arxiv.org/abs/2309.08532))

#### Further changes:
* Added example classification datasets used in the [EvoPrompt paper](https://arxiv.org/abs/2309.08532)
* Added dummy classes for testing
* Added example scripts and configs for experiments
* Added experiment results and evaluation notebooks

**Full Changelog**: [here](https://github.com/finitearth/promptolution/commits/0.1.0)

