# Release Notes

## Release v1.1.1
### What's Changed
- deleted poetry.lock
- updated transformers dependency: bumped from 4.46.3 to 4.48.0 

## Release v1.1.0
### What's changed
#### Added features
* Enable reading tasks from a pandas dataframe

#### Further Changes:
* deleted experiment files from the repo folders (logs, configs, etc.)
* improved opros meta-prompt
* added support for python versions from 3.9 onwards (previously 3.11)

## Release v1.0.1
### What's changed
#### Added features
-

#### Further Changes:
* fixed release notes

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

