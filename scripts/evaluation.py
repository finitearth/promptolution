from logging import INFO, Logger
import argparse

from promptolution.callbacks import LoggerCallback, CSVCallback, BestPromptCallback
from promptolution.llm import get_llm
from promptolution.config import Config
from promptolution.optimizer import get_optimizer
from promptolution.predictor import get_predictor
from promptolution.tasks import get_tasks

logger = Logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dummy.ini")
    return parser.parse_args()

def main(args):
    config = Config(args.config)
    task = get_tasks(config)[0]
    logger.critical(f"Task: {task.description}")

    predictor = get_predictor(config, classes=task.classes)
    best_prompt_callback = BestPromptCallback()
    callbacks = [
        LoggerCallback(logger),
        CSVCallback(config.logging_dir),
        best_prompt_callback
    ]
    prompt_template = open(config.meta_prompt_path, "r").read()
    if config.include_task_desc:
        prompt_template = prompt_template.replace("<task_desc>", task.description)

    if "local" in config.meta_llm:
        meta_llm = get_llm(config.meta_llm, batch_size=config.meta_bs)
    else:
        meta_llm = get_llm(config.meta_llm)

    optimizer = get_optimizer(
        config,
        meta_llm=meta_llm,
        task=task,
        initial_prompts=task.initial_population,
        callbacks=callbacks,
        prompt_template=prompt_template,
        predictor=predictor,
    )
    
    logger.critical("🚨🚨🚨HEREEE WEEEE GOOO🚨🚨🚨")
    optimizer.optimize(config.n_steps)
    logger.critical("🎉We did it🥳")
    best_prompt, best_score = best_prompt_callback.get_best_prompt()
    logger.critical(f"Final prompt: {best_prompt}, with score: {best_score}")
    # TODO evaluate final prompt on test data split
    # evaluation on test data
    test_score = task.evaluate(best_prompt, predictor)
    logger.critical(f"Test score: {test_score}")


if __name__ == "__main__":
    args = parse_args()
    main(args)