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
    llm_name = "vllm-Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"

    config = Config(
        meta_llm=llm_name,
        ds_path="data_sets/cls/agnews",
        task_name="agnews",
        n_steps=10,
        optimizer="opro",
        downstream_llm=llm_name,
        evaluation_llm=llm_name,

    )
    task = get_task(config, split="dev")

    llm = get_llm(
        config.meta_llm,
        max_model_len=2000,
        model_storage_path="../models/",
        revision="main"
    )
    predictor = get_predictor(llm, classes=task.classes)
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
