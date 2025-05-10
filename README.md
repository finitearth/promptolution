![promptolution](https://github.com/user-attachments/assets/84c050bd-61a1-4f2e-bc4e-874d9b4a69af)

[![CI](https://github.com/finitearth/promptolution/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/finitearth/promptolution/actions/workflows/ci.yml)
[![Docs](https://github.com/finitearth/promptolution/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/finitearth/promptolution/actions/workflows/docs.yml)

# Promptolution
Promptolution is a library that provides a modular and extensible framework for implementing prompt tuning experiments. It offers a user-friendly interface to assemble the core components for various prompt optimization tasks.

In addition, this repository contains our experiments for the paper "Towards Cost-Effective Prompt Tuning: Evaluating the Effects of Model Size, Model Family and Task Descriptions in EvoPrompt".

This project was developed by [Timo Heiß](https://www.linkedin.com/in/timo-heiss/), [Moritz Schlager](https://www.linkedin.com/in/moritz-schlager/) and [Tom Zehle](https://www.linkedin.com/in/tom-zehle/).

## Getting Started
Take a look at our getting started notebook: [getting_started.ipynb](https://github.com/finitearth/promptolution/blob/main/notebooks/getting_started.ipynb)

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

## Documentation

A comprehensive documentation with API reference is availabe at https://finitearth.github.io/promptolution/.

## Usage

Create API Keys for the models you want to use:
- OpenAI: store token in openaitoken.txt
- Anthropic: store token in anthropictoken.txt
- DeepInfra (for Llama): store token in deepinfratoken.txt

## Optimization Algorithms to choose from
| **Name** | **# init population** | **Exploration** | **Costs** | **Convergence Speed** | **Parallelizable** | **Utilizes Failure Cases** |
|:--------:|:---------------------:|:---------------:|:---------:|:---------------------:|:------------------:|:---------------------:|
| EvoPrompt DE | 8-12 | 👍 | 💲 | ⚡⚡ | ✅ | ❌ |
| EvoPrompt GA | 8-12 | 👍 | 💲 | ⚡⚡ | ✅ | ❌ |
| OPro | 0 | 👎 | 💲💲 | ⚡ | ❌ | ❌ |

## Core Components

- Task: Encapsulates initial prompts, dataset features, targets, and evaluation methods.
- Predictor: Implements the prediction logic, interfacing between the Task and LLM components.
- LLM: Unifies the process of obtaining responses from language models, whether locally hosted or accessed via API.
- Optimizer: Implements prompt optimization algorithms, utilizing the other components during the optimization process.
- Exemplar Selectors: Implements algorithms for the search of few shot examples that are added to the prompt.

## Key Features

- Modular and object-oriented design
- Extensible architecture
- Easy-to-use interface for assembling experiments
- Parallelized LLM requests for improved efficiency
- Integration with langchain for standardized LLM API calls
- Detailed logging and callback system for optimization analysis



This project was developed for the "AutoML in the age of large pre-trained models" seminar at LMU Munich.
