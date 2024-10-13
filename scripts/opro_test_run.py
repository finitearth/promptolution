"""Test run for the Opro optimizer."""

<<<<<<< HEAD
=======
from configparser import ConfigParser
>>>>>>> main
from logging import Logger

from promptolution.callbacks import LoggerCallback
from promptolution.llms import get_llm
from promptolution.optimizers import Opro
from promptolution.predictors import get_predictor
<<<<<<< HEAD
from promptolution.tasks import get_task

from promptolution.config import Config
=======
from promptolution.tasks import get_tasks
>>>>>>> main

logger = Logger(__name__)


def main():
    """Run a test run for the Opro optimizer."""
<<<<<<< HEAD
    config = Config(
        meta_llm="meta-llama/Meta-Llama-3-8B-Instruct",
        ds_path="data_sets/agnews",
        task_name="agnews",
        n_steps=10,
        optimizer="opro",
        downstream_llm="meta-llama/Meta-Llama-3-8B-Instruct",
        evaluation_llm="meta-llama/Meta-Llama-3-8B-Instruct",

    )
    task = get_task(config, split="dev")
    predictor = get_predictor(config.evaluation_llm, classes=task.classes)

    llm = get_llm(config.meta_llm)
=======
    config = ConfigParser()
    config.task_name = "agnews"
    config.ds_path = "data_sets/cls/agnews"
    config.random_seed = 42

    llm = get_llm("meta-llama/Meta-Llama-3-8B-Instruct")
    task = get_tasks(config)[0]
    predictor = get_predictor("meta-llama/Meta-Llama-3-8B-Instruct", classes=task.classes)

>>>>>>> main
    optimizer = Opro(
        llm,
        initial_prompts=task.initial_population,
        task=task,
        predictor=predictor,
        callbacks=[LoggerCallback(logger)],
        n_samples=5,
    )
    prompts = optimizer.optimize(n_steps=10)

    logger.info(f"Optimized prompts: {prompts}")


if __name__ == "__main__":
    main()
