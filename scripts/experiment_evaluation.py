"""Experiment for paper Towards Cost-Effective Prompt Tuning to evaluate best prompts."""
from argparse import ArgumentParser
from configparser import ConfigParser
from logging import INFO, Logger
from pathlib import Path
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from promptolution.config import Config
from promptolution.predictors import get_predictor
from promptolution.tasks import get_tasks

logger = Logger(__name__)
logger.setLevel(INFO)


def get_best_prompt_from_csv(csv_path: str) -> Tuple[str, float]:
    """Get the best prompt from a csv file.

    This function gets the best prompst from a csv file by looking at the most recent step
    and picking the prompt with the highest score in that step. There are multiple occurrences of the same step,
    we only pick the highest score (first occurrrence if there are multiple).
    """
    df = pd.read_csv(csv_path)
    max_step = df["step"].max()
    df = df[df["step"] == max_step]
    best_score = df["score"].max()
    best_prompt = df[df["score"] == best_score]["prompt"].values[0]
    return best_prompt, best_score


def evaluate_best_prompts(
    experiment_name: str, target_experiment: str, logging_dir: Path, downstream_llm: str, n_samples: int, seed: int
) -> pd.DataFrame:
    """Evaluate the best prompts from a csv file."""
    # convert the logging directory to a string
    logging_dir = str(logging_dir)

    best_prompt, best_score = get_best_prompt_from_csv(logging_dir)
    # extract information about the experiment from the filename
    logging_dir = logging_dir.replace(f"logs\\{target_experiment}\\", "")
    logging_dir = logging_dir.replace(".csv", "")

    task_name, optimizer, meta_llm, evaluation_llm, random_seed = logging_dir.split("_")

    # create config for the experiment
    config = Config(
        task_name=task_name,
        ds_path=f"data_sets/cls/{task_name}",
        random_seed=seed,
    )

    # create a test task to retrieve the samples to evaluate
    test_task = get_tasks(config, split="test")[0]
    test_predictor = get_predictor(downstream_llm, classes=test_task.classes)

    # evaluate the best prompt on the test set
    test_score = test_task.evaluate(best_prompt, test_predictor, subsample=True, n_samples=n_samples)

    # save the test score to a csv file
    df = pd.DataFrame(
        {
            "task": task_name,
            "optimizer": optimizer,
            "meta_llm": meta_llm,
            "downstream_llm": downstream_llm,
            "evaluation_llm": evaluation_llm,
            "random_seed": random_seed,
            "best_prompt": best_prompt,
            "dev_score": best_score,
            "test_score": test_score,
        },
        index=[0],
    )
    df.to_csv(f"logs/{experiment_name}/best_scores.csv", mode="a", header=False, index=False)


def main():
    """Run the evaluation of best prompts."""
    # read experiments
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-e", "--experiment", type=str, help="Experiment Config Filepath")
    args = arg_parser.parse_args()
    all_configs = ConfigParser()
    all_configs.read(args.experiment)
    print(all_configs)

    experiment_name = all_configs["experiment"]["name"]
    target_experiment = all_configs["target_experiment"]["name"]
    downstream_llms = all_configs["downstream_llm"]["name"].split(",")
    tasks = all_configs["task"]["task_name"].split(",")
    optimizers = all_configs["task"]["optimizer"].split(",")
    llms = all_configs["task"]["llm"].split(",")
    seed = int(all_configs["task"]["subsample_seed"])
    n_samples = int(all_configs["task"]["n_samples"])

    logger.critical(f"Starting evaluation of best prompts for experiment {target_experiment}")

    for downstream_llm in downstream_llms:
        logger.critical(f"Downstream LLM: {downstream_llm}")
        # iterate through all files in the target experiment folder
        for logging_dir in tqdm(Path(f"logs/{target_experiment}").rglob("*.csv")):
            if (
                "best_scores" in str(logging_dir)
                or not any(task in str(logging_dir) for task in tasks)
                or not any(optimizer in str(logging_dir) for optimizer in optimizers)
                or not any(llm in str(logging_dir) for llm in llms)
            ):
                continue

            print(logging_dir)
            # extract the logging directory from the file path by removing the directory and experiment name
            evaluate_best_prompts(
                experiment_name=experiment_name,
                target_experiment=target_experiment,
                logging_dir=logging_dir,
                downstream_llm=downstream_llm,
                n_samples=n_samples,
                seed=seed,
            )

    logger.critical("Evaluation of best prompts finished.")


if __name__ == "__main__":
    main()
