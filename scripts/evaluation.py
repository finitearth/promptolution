from configparser import ConfigParser
from logging import INFO, Logger

from promptolution.callbacks import LoggerCallback
from promptolution.llm import get_llm
from promptolution.optimizer import get_optimizer
from promptolution.predictor import get_predictor
from promptolution.tasks import get_tasks

if __name__ == "__main__":
    logger = Logger(__name__)
    logger.setLevel(INFO)

    config = ConfigParser()
    config.read("configs/test.ini")
    tasks = get_tasks(config)
    for task in tasks:
        logger.critical(f"Task: {task.description}")
        logger.critical("🚨🚨🚨HEREEE WEEEE GOOO🚨🚨🚨")

        predictor = get_predictor(config["downstream_llms"]["names"])#Predictor()
        callbacks = [LoggerCallback(logger)]
        prompt_template = open(config["optimizer"]["meta_prompt_path"], "r").read()
        meta_llm = get_llm(config["meta_llms"]["names"])#APILLM(config["meta_llms"]["names"])#APILLM(config["meta_llms"]["names"])
        optimizer = get_optimizer(
            config["optimizer"]["name"],
            meta_llm=meta_llm,
            task=task,
            initial_prompts=task.initial_population,
            callbacks=callbacks,
            prompt_template=prompt_template,
            predictor=predictor,
        )
        optimizer.optimize(int(config["tasks"]["steps"]))
        # TODO evaluate final prompt on test data split
        logger.critical(f"Final prompt: {optimizer.prompts[0]}")
        # evaluation on test data
        test_score = task.evaluate(optimizer.prompts[0], predictor, seed=42)