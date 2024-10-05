"""Experiments for paper Towards Cost-Effective Prompt Tuning initial prompt evaluation."""
from argparse import ArgumentParser
from configparser import ConfigParser
from logging import INFO, Logger
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from promptolution.config import Config
from promptolution.predictors import get_predictor
from promptolution.tasks import get_task

logger = Logger(__name__)
logger.setLevel(INFO)


def evaluate_prompts(
    experiment_name: str,
    downstream_llm: str,
    task_name: str,
    random_seed: int,
    n_samples: int,
    prompts: List[str],
) -> pd.DataFrame:
    """Evaluate the best prompts from a csv file."""
    # create config for the experiment
    config = Config(
        task_name=task_name,
        ds_path=f"data_sets/{task_name}",
        random_seed=random_seed,
    )

    # create a test task to retrieve the samples to evaluate
    test_task = get_task(config.ds_path, split="test", random_seed=config.random_seed, task_name=config.task_name)
    test_predictor = get_predictor(downstream_llm, classes=test_task.classes)

    # evaluate the best prompt on the test set
    for prompt in prompts:
        # check if the combination of task, downstream_llm, seed and prompt has already been evaluated and is a row in
        # the csv file
        if Path(f"logs/{experiment_name}/best_scores.csv").exists():
            df = pd.read_csv(f"logs/{experiment_name}/best_scores.csv")
            if (
                len(
                    df[
                        (df["task"] == task_name)
                        & (df["downstream_llm"] == downstream_llm)
                        & (df["random_seed"] == random_seed)
                        & (df["prompt"] == prompt)
                    ]
                )
                > 0
            ):
                print(f"Prompt for task {task_name} has already been evaluated.")
                continue

            else:
                test_score = test_task.evaluate(prompt, test_predictor, subsample=True, n_samples=n_samples)
                # save the test score to a csv file
                df = pd.DataFrame(
                    {
                        "task": task_name,
                        "downstream_llm": downstream_llm,
                        "random_seed": random_seed,
                        "prompt": prompt,
                        "test_score": test_score,
                    },
                    index=[0],
                )
                df.to_csv(f"logs/{experiment_name}/best_scores.csv", mode="a", header=False, index=False)


def main():
    """Run experiment."""
    # read experiments
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-e", "--experiment", type=str, help="Experiment Config Filepath")
    args = arg_parser.parse_args()
    all_configs = ConfigParser()
    all_configs.read(args.experiment)
    print(all_configs)

    experiment_name = all_configs["experiment"]["name"]
    downstream_llms = all_configs["downstream_llm"]["name"].split(",")
    tasks = all_configs["task"]["task_name"].split(",")
    seed = int(all_configs["task"]["subsample_seed"])
    n_prompts = int(all_configs["task"]["n_prompts"])
    n_samples = int(all_configs["task"]["n_samples"])

    logger.critical(f"Starting evaluation of best prompts for experiment {experiment_name}")

    for downstream_llm in downstream_llms:
        logger.critical(f"Downstream LLM: {downstream_llm}")

        for task in tasks:
            # sample initial prompt (read txt from datasets/cls/task_name/prompts.txt)
            task_path = Path(f"data_sets/{task}/prompts.txt")
            with open(task_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            lines = [line.strip() for line in lines]
            # sample n_prompts prompts with seed
            np.random.seed(seed)
            prompts = np.random.choice(lines, n_prompts, replace=False)

            evaluate_prompts(
                experiment_name=experiment_name,
                downstream_llm=downstream_llm,
                task_name=task,
                random_seed=seed,
                n_samples=n_samples,
                prompts=prompts,
            )
            print(task)

    logger.critical("Evaluation of best prompts finished.")


if __name__ == "__main__":
    main()
