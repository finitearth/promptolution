"""Test run for the Opro optimizer."""
import argparse
from logging import Logger

from promptolution.callbacks import LoggerCallback, FileOutputCallback
from promptolution.helpers import run_optimization

from promptolution.config import Config

logger = Logger(__name__)

"""Run a test run for any of the implemented optimizers."""
parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--model-storage-path", default="../models/")
parser.add_argument("--optimizer", default="evopromptde")
parser.add_argument("--n-steps", type=int, default=10)
parser.add_argument("--token", default=None)
parser.add_argument("--seed", type=int, default=187)
args = parser.parse_args()

config = Config(
    meta_llm=args.model,
    ds_path="data_sets/cls/agnews",
    task_name="agnews",
    predictor="FirstOccurenceClassificator",
    n_steps=args.n_steps,
    optimizer=args.optimizer,
    downstream_llm=args.model,
    evaluation_llm=args.model,
    api_token=args.token,
    model_storage_path=args.model_storage_path,
    random_seed=args.seed,
)

prompts = run_optimization(config, callbacks=[LoggerCallback(logger), FileOutputCallback(f"results/seedingtest/{args.model}/", "csv")], use_token=not args.token is None)

logger.info(f"Optimized prompts: {prompts}")
