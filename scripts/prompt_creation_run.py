"""Script to run prompt creation and evaluation."""

from configparser import ConfigParser
from logging import Logger

from promptolution.llms import get_llm
from promptolution.predictors import get_predictor
from promptolution.tasks import get_task
from promptolution.utils.prompt_creation import create_prompt_variation, create_prompts_from_samples

logger = Logger(__name__)


def main():
    """Main function to run the experiment."""
    config = ConfigParser()
    config.task_name = "agnews"
    config.ds_path = "data_sets/agnews"
    config.random_seed = 42

    llm = get_llm("meta-llama/Meta-Llama-3-8B-Instruct")
    task = get_task(config, split="dev")

    predictor = get_predictor(llm, classes=task.classes)

    init_prompts = create_prompts_from_samples(task, llm)
    logger.critical(f"Initial prompts: {init_prompts}")

    # evaluate on task
    scores = task.evaluate(init_prompts, predictor)
    logger.critical(f"Initial scores {scores.mean()} +/- {scores.std()}")

    varied_prompts = create_prompt_variation(init_prompts, llm)[0]

    logger.critical(f"Varied prompts: {varied_prompts}")

    # evaluate on task
    scores = task.evaluate(varied_prompts, predictor)
    logger.critical(f"Varied scores {scores.mean()} +/- {scores.std()}")


if __name__ == "__main__":
    main()
