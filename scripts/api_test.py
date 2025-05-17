"""Test run for the Opro optimizer."""

import argparse
import random
from logging import Logger

from promptolution.callbacks import LoggerCallback
from promptolution.templates import EVOPROMPT_GA_TEMPLATE
from promptolution.tasks import ClassificationTask
from promptolution.predictors import MarkerBasedClassifier
from promptolution.optimizers import EvoPromptGA
from datasets import load_dataset

from promptolution.llms.api_llm import APILLM

logger = Logger(__name__)

"""Run a test run for any of the implemented optimizers."""
parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="https://api.openai.com/v1")
parser.add_argument("--model", default="gpt-4o-2024-08-06")
# parser.add_argument("--base-url", default="https://api.deepinfra.com/v1/openai")
# parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
# parser.add_argument("--base-url", default="https://api.anthropic.com/v1/")
# parser.add_argument("--model", default="claude-3-haiku-20240307")
parser.add_argument("--n-steps", type=int, default=2)
parser.add_argument("--token", default=None)
args = parser.parse_args()

df = load_dataset("SetFit/ag_news", split="train", revision="main").to_pandas().sample(300)

df["input"] = df["text"]
df["target"] = df["label_text"]

task = ClassificationTask(
    df,
    description="The dataset contains news articles categorized into four classes: World, Sports, Business, and Tech. The task is to classify each news article into one of the four categories.",
    x_column="input",
    y_column="target",
)

initial_prompts = [
    "Classify this news article as World, Sports, Business, or Tech. Provide your answer between <final_answer> and </final_answer> tags.",
    "Read the following news article and determine which category it belongs to: World, Sports, Business, or Tech. Your classification must be placed between <final_answer> </final_answer> markers.",
    "Your task is to identify whether this news article belongs to World, Sports, Business, or Tech news. Provide your classification between the markers <final_answer> </final_answer>.",
    "Conduct a thorough analysis of the provided news article and classify it as belonging to one of these four categories: World, Sports, Business, or Tech. Your answer should be presented within <final_answer> </final_answer> markers.",
]

llm = APILLM(api_url=args.base_url, model_id=args.model, api_key=args.token)
downstream_llm = llm
meta_llm = llm

predictor = MarkerBasedClassifier(downstream_llm, classes=task.classes)

callbacks = [LoggerCallback(logger)]

optimizer = EvoPromptGA(
    task=task,
    prompt_template=EVOPROMPT_GA_TEMPLATE,
    predictor=predictor,
    meta_llm=meta_llm,
    initial_prompts=initial_prompts,
    callbacks=callbacks,
    n_eval_samples=20,
)

best_prompts = optimizer.optimize(n_steps=args.n_steps)

logger.info(f"Optimized prompts: {best_prompts}")
