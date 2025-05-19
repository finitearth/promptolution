![promptolution](https://github.com/user-attachments/assets/84c050bd-61a1-4f2e-bc4e-874d9b4a69af)

![Coverage](https://img.shields.io/badge/Coverage-87%25-green)
[![CI](https://github.com/finitearth/promptolution/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/finitearth/promptolution/actions/workflows/ci.yml)
[![Docs](https://github.com/finitearth/promptolution/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/finitearth/promptolution/actions/workflows/docs.yml)
![Code Style](https://img.shields.io/badge/Code%20Style-black-black)
![Python Versions](https://img.shields.io/badge/Python%20Versions-≥3.9-blue)



Promptolution is a library that provides a modular and extensible framework for implementing prompt tuning for single tasks and larger experiments. It offers a user-friendly interface to assemble the core components for various prompt optimization tasks.

This project was developed by [Timo Heiß](https://www.linkedin.com/in/timo-heiss/), [Moritz Schlager](https://www.linkedin.com/in/moritz-schlager/) and [Tom Zehle](https://www.linkedin.com/in/tom-zehle/) as part of a study program at LMU Munich.

## Installation

Use pip to install our library:

```
pip install promptolution[api]
```

If you want to run your prompt optimization locally, either via transformers or vLLM, consider running:

```
pip install promptolution[vllm,transformers]
```

Alternatively, clone the repository, run

```
poetry install
```

to install the necessary dependencies. You might need to install [pipx](https://pipx.pypa.io/stable/installation/) and [poetry](https://python-poetry.org/docs/) first.

## Usage

To get started right a way take a look at our [getting started notebook](https://github.com/finitearth/promptolution/blob/main/notebooks/getting_started.ipynb).
For more details a comprehensive **documentation** with API reference is availabe at https://finitearth.github.io/promptolution/.

### Featured Optimizers

|   **Name**    |                    **Paper**                     | **init prompts** | **Exploration** | **Costs**  | **Parallelizable** | **Utilizes Fewshot Examples** |
| :-----------: | :----------------------------------------------: | :--------------: | :-------------: | :-------: | :-------------------: | :---------------------------: |
|    `CAPO`     | [Zehle et al.](https://arxiv.org/abs/2504.16005) |    _required_    |       👍        |    💲     |        ✅         |              ✅               |
| `EvoPromptDE` |  [Guo et al.](https://arxiv.org/abs/2309.08532)  |    _required_    |       👍        |   💲💲    |               ✅         |              ❌               |
| `EvoPromptGA` |  [Guo et al.](https://arxiv.org/abs/2309.08532)  |    _required_    |       👍        |   💲💲    |               ✅         |              ❌               |
|    `OPRO`     | [Yang et al.](https://arxiv.org/abs/2309.03409)  |    _optional_    |       👎        |   💲💲    |                  ❌         |              ❌               |

### Core Components

- Task: Encapsulates initial prompts, dataset features, targets, and evaluation methods.
- Predictor: Implements the prediction logic, interfacing between the Task and LLM components.
- LLM: Unifies the process of obtaining responses from language models, whether locally hosted or accessed via API.
- Optimizer: Implements prompt optimization algorithms, utilizing the other components during the optimization process.

### Key Features

- Modular and object-oriented design
- Extensible architecture
- Easy-to-use interface for assembling experiments
- Parallelized LLM requests for improved efficiency
- Integration with langchain for standardized LLM API calls
- Detailed logging and callback system for optimization analysis

## Changelog

https://finitearth.github.io/promptolution/release-notes/

## Contributing

The first step to contributing is to open an issue describing the bug, feature or enhancements. Ensure the issue is clearly described, assigned and properly tagged. All work should be linked to an open issue.

### Code Style and Linting

We use Black for code formatting, Flake8 for linting, pydocstyle for docstring conventions (Google format), and isort to sort imports. All these checks are enforced via pre-commit hooks, which automatically run on every commit. Install the pre-commit hooks to ensure that all checks run automatically:

```
pre-commit install
```

To run all checks manually:

```
pre-commit run --all-files
```

### Branch Protection and Merging Guidelines

- The main branch is protected. No direct commits are allowed for non-administrators.
- Rebase your branch on main before opening a pull request.
- All contributions must be made on dedicated branches linked to specific issues.
- Name the branch according to {prefix}/{description} with one of the prefixes fix, feature, chore, or refactor.
- A pull request must have at least one approval from a code owner before it can be merged into main.
- CI checks must pass before a pull request can be merged.
- New releases will only be created by code owners.

### Testing

We use pytest to run tests, and coverage to track code coverage. Tests automatically run on pull requests and pushes to the main branch, but please ensure they also pass locally before pushing!
To run the tests with coverage locally, use the following commands or your IDE's test runner:

```
poetry run python -m coverage run -m pytest
```

To see the coverage report run:
```
poetry run python -m coverage report
```
