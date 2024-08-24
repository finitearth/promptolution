# Promptolution

Project for seminar "AutoML in the age of large pre-trained language models" at LMU Munich , developed by [Timo Heiß](https://www.linkedin.com/in/timo-heiss/), [Moritz Schlager](https://www.linkedin.com/in/moritz-schlager/) and [Tom Zehle](https://www.linkedin.com/in/tom-zehle/).

## Set Up

After having cloned the repository, run

```
poetry install
```

to install the necessary dependencies.

You might need to install [pipx](https://pipx.pypa.io/stable/installation/) and [poetry](https://python-poetry.org/docs/) first.

## Usage

Create API Keys for the models you want to use:
- OpenAI: store token in openaitoken.txt
- Anthropic: store token in anthropictoken.txt
- DeepInfra (for Llama): store token in deepinfratoken.txt

TODO: CREATE WHEEL FILES AND UPLOAD TO PIP

Run experiments based on config via:

```
poetry run python scripts/experiment_runs.py --experiment "configs/<my_experiment>.ini"
```
where `<my_experiment>.ini` is a config based on our templates.