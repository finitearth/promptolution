from promptolution.llms import get_llm
from promptolution.tasks import get_tasks
from promptolution.predictors import get_predictor
from promptolution.optimizers import Opro
from promptolution.callbacks import LoggerCallback

from configparser import ConfigParser
from logging import Logger


logger = Logger(__name__)

def main():
    config = ConfigParser()
    config.task_name = "agnews"
    config.ds_path = "data_sets/cls/agnews"
    config.random_seed = 42

    llm = get_llm("meta-llama/Meta-Llama-3-8B-Instruct")
    task = get_tasks(config)[0]
    predictor = get_predictor("meta-llama/Meta-Llama-3-8B-Instruct", classes=task.classes)

    optimizer = Opro(llm, initial_prompts=task.initial_population, task=task, predictor=predictor, callbacks=[LoggerCallback(logger)], n_samples=5)
    prompts = optimizer.optimize(n_steps=10)

    logger.info(f"Optimized prompts: {prompts}")

if __name__ == "__main__":
    main()

