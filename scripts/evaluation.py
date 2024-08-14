from configparser import ConfigParser
from logging import INFO, Logger

from promptolution.callbacks import LoggerCallback
from promptolution.llm import DummyLLM
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

        predictor = get_predictor("dummy", task)
        callbacks = [LoggerCallback(logger)]
        prompt_template = open(config["optimizer"]["meta_prompt_path"], "r").read()
        meta_llm = DummyLLM()
        optimizer = get_optimizer(
            config["optimizer"]["name"],
            meta_llm=meta_llm,
            task=task,
            initial_prompts=["Roaldn", "Dahl", "is", "a", "famous", "author", "mamamia"],
            callbacks=callbacks,
            prompt_template=prompt_template,
            predictor=predictor,
        )
        for _ in range(int(config["tasks"]["steps"])):
            optimizer.step()
