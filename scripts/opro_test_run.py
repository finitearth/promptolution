"""Test run for the Opro optimizer."""

from logging import Logger

from promptolution.callbacks import LoggerCallback
from promptolution.llms import get_llm
from promptolution.optimizers import Opro
from promptolution.predictors import get_predictor
from promptolution.tasks import get_task

from promptolution.config import Config

logger = Logger(__name__)


def main():
    """Run a test run for the Opro optimizer."""
    config = Config(
        meta_llm="meta-llama/Meta-Llama-3-8B-Instruct",
        ds_path="data_sets/agnews",
        task_name="agnews",
        n_steps=10,
        optimizer="opro",
        meta_prompt_path="promptolution/templates/opro_template.txt",
        downstream_llm="meta-llama/Meta-Llama-3-8B-Instruct",
        evaluation_llm="meta-llama/Meta-Llama-3-8B-Instruct",

    )
    task = get_task(config, split="test")
    predictor = get_predictor(config.evaluation_llm, classes=task.classes)

    llm = get_llm(config.meta_llm)
    optimizer = Opro(
        llm,
        initial_prompts=task.initial_population,
        task=task,
        predictor=predictor,
        callbacks=[LoggerCallback(logger)],
        n_samples=5,
        template_path=config.meta_prompt_path,
    )
    prompts = optimizer.optimize(n_steps=10)

    logger.info(f"Optimized prompts: {prompts}")


if __name__ == "__main__":
    main()
