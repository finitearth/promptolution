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
        meta_llm="vllm-shuyuej/Llama-3.3-70B-Instruct-GPTQ",
        ds_path="data_sets/cls/agnews",
        task_name="agnews",
        n_steps=10,
        optimizer="opro",
        downstream_llm="vllm-shuyuej/Llama-3.3-70B-Instruct-GPTQ",
        evaluation_llm="vllm-shuyuej/Llama-3.3-70B-Instruct-GPTQ",

    )
    task = get_task(config, split="dev")
    predictor = get_predictor(config.evaluation_llm, classes=task.classes)

    llm = get_llm(config.meta_llm)
    optimizer = Opro(
        llm,
        initial_prompts=task.initial_population,
        task=task,
        predictor=predictor,
        callbacks=[LoggerCallback(logger)],
        n_samples=5,
    )
    prompts = optimizer.optimize(n_steps=2)

    logger.info(f"Optimized prompts: {prompts}")


if __name__ == "__main__":
    main()
