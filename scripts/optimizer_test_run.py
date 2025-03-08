"""Test run for the Opro optimizer."""
import argparse
from logging import Logger

from promptolution.callbacks import LoggerCallback
from promptolution.llms import get_llm
from promptolution.optimizers import Opro, EvoPromptDE, EvoPromptGA
from promptolution.predictors import get_predictor
from promptolution.tasks import get_task

from promptolution.config import Config

logger = Logger(__name__)

"""Run a test run for any of the implemented optimizers."""
parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--model-storage-path", default="../models/")
parser.add_argument("--optimizer", default="evoprompt_de")
parser.add_argument("--n-steps", default=10)
args = parser.parse_args()

config = Config(
    meta_llm=args.model,
    ds_path="data_sets/cls/agnews",
    task_name="agnews",
    n_steps=10,
    optimizer="opro",
    downstream_llm=args.model,
    evaluation_llm=args.model,

)
task = get_task(config, split="dev")

llm = get_llm(
    config.meta_llm,
    max_model_len=2000,
    model_storage_path=args.model_storage_path,
    revision="main"
)
predictor = get_predictor(llm, classes=task.classes)

if args.optimizer == "evoprompt_de":
    optimizer = EvoPromptDE(
        meta_llm=llm,
        initial_prompts=task.initial_population,
        task=task,
        predictor=predictor,
        callbacks=[LoggerCallback(logger)],
        n_samples=5,
    )
elif args.optimizer == "evoprompt_ga":
    optimizer = EvoPromptGA(
        meta_llm=llm,
        initial_prompts=task.initial_population,
        task=task,
        predictor=predictor,
        callbacks=[LoggerCallback(logger)],
        n_samples=5,
    )
else:
    optimizer = Opro(
        meta_llm=llm,
        initial_prompts=task.initial_population,
        task=task,
        predictor=predictor,
        callbacks=[LoggerCallback(logger)],
        n_samples=5,
    )

prompts = optimizer.optimize(n_steps=args.n_steps)

logger.info(f"Optimized prompts: {prompts}")
