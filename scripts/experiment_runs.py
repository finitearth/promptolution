from configparser import ConfigParser
from argparse import ArgumentParser
from logging import INFO, Logger
from pathlib import Path

import numpy as np
import pandas as pd

from promptolution.callbacks import BestPromptCallback, CSVCallback, ProgressBarCallback
from promptolution.config import Config
from promptolution.llms import get_llm
from promptolution.optimizers import get_optimizer
from promptolution.predictors import get_predictor
from promptolution.tasks import get_tasks

logger = Logger(__name__)
logger.setLevel(INFO)


def main():
    # read experiments ini
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-e', '--experiment', type=str, help='Experiment Config Filepath')
    args = arg_parser.parse_args()
    all_configs = ConfigParser()
    all_configs.read(args.experiment)
    experiment_name = all_configs["experiment"]["name"]
    task_names = all_configs["task"]["task_name"].split(",")
    meta_prompt_paths = all_configs["optimizer"]["meta_prompt_path"].split(",")
    optimizer_names = all_configs["optimizer"]["name"].split(",")
    meta_llms = all_configs["meta_llm"]["name"].split(",")
    evaluator_llms = all_configs["evaluator_llm"]["name"].split(",")
    downstream_llms = all_configs["downstream_llm"]["name"].split(",")
    for task_name in task_names:
        for optimizer_name, meta_prompt_path in zip(optimizer_names, meta_prompt_paths):
            for evaluator_llm, meta_llm in zip(evaluator_llms, meta_llms):
                for downstream_llm in downstream_llms:
                    for random_seed in [42, 47, 69]:
                        config = Config(
                            task_name=task_name,
                            ds_path=f"data_sets/cls/{task_name}",
                            n_steps=int(all_configs["task"]["steps"]),
                            optimizer=optimizer_name,
                            meta_llm=meta_llm,
                            downstream_llm=downstream_llm,
                            meta_prompt_path=meta_prompt_path,
                            init_pop_size=int(all_configs["optimizer"]["init_population"]),
                            logging_dir=f"logs/{experiment_name}/{task_name}_{optimizer_name}_{meta_llm}_{evaluator_llm}_{random_seed}.csv",
                            include_task_desc=False,
                            random_seed=random_seed,
                            evaluation_llm=evaluator_llm,
                            selection_mode="random",
                            donor_random=False,
                        )
                        # skip already performed experiments
                        if Path(config.logging_dir).exists():
                            continue
                        run_experiment(config)


def run_experiment(config: Config):
    task = get_tasks(config)[0]
    init_populations = task.initial_population
    # subsample using random seed
    np.random.seed(config.random_seed)
    init_population = np.random.choice(init_populations, config.init_pop_size, replace=False)
    logger.critical(f"Task: {task.description}")

    predictor = get_predictor(config.downstream_llm, classes=task.classes)
    best_prompt_callback = BestPromptCallback()
    callbacks = [
        # LoggerCallback(logger),
        CSVCallback(config.logging_dir),
        best_prompt_callback,
        ProgressBarCallback(config.n_steps),
    ]
    prompt_template = open(config.meta_prompt_path, "r").read()
    if config.include_task_desc:
        prompt_template = prompt_template.replace("<task_desc>", task.description)  # TODO how to predict in evaluate

    if "local" in config.meta_llm:
        meta_llm = get_llm(config.meta_llm, batch_size=config.meta_bs)
    else:
        meta_llm = get_llm(config.meta_llm)

    optimizer = get_optimizer(
        config,
        meta_llm=meta_llm,
        task=task,
        initial_prompts=init_population,
        callbacks=callbacks,
        prompt_template=prompt_template,
        predictor=predictor,
    )

    logger.critical("🚨🚨🚨HEREEE WEEEE GOOO🚨🚨🚨")
    optimizer.optimize(config.n_steps)
    logger.critical("🎉We did it🥳")
    best_prompt, best_score = best_prompt_callback.get_best_prompt()
    logger.critical(f"Final prompt: {best_prompt}, with score: {best_score}")

    eval_task = get_tasks(config, split="test")[0]
    eval_predictor = get_predictor(config.evaluation_llm, classes=eval_task.classes)
    test_score = eval_task.evaluate(best_prompt, eval_predictor, subsample=False)

    # save test score to csv
    df = pd.DataFrame(
        {
            "task": config.task_name,
            "optimizer": config.optimizer,
            "meta_llm": config.meta_llm,
            "downstream_llm": config.downstream_llm,
            "evaluation_llm": config.evaluation_llm,
            "random_seed": config.random_seed,
            "test_score": test_score,
        },
    )
    df.to_csv("logs/best_scores.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
