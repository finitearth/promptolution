"""Test run for the Opro optimizer."""


import argparse
import random
from logging import Logger

from datasets import load_dataset

from promptolution.llms import APILLM
from promptolution.optimizers import EVOPROMPT_GA_TEMPLATE, EvoPromptGA
from promptolution.predictors import MarkerBasedClassifier
from promptolution.tasks import ClassificationTask
from promptolution.utils import FileOutputCallback, LoggerCallback, TokenCountCallback

logger = Logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--output-dir", default="results/evoprompt_ga_test/")
parser.add_argument("--n-steps", type=int, default=2)
parser.add_argument("--base_url", default="https://api.deepinfra.com/v1/openai")
parser.add_argument("--api_url", default=None)
args = parser.parse_args()

callbacks = [
    LoggerCallback(logger),
    FileOutputCallback(args.output_dir, file_type="csv"),
    TokenCountCallback(100000, "input_tokens"),
]

df = load_dataset("SetFit/ag_news", split="train", revision="main").to_pandas().sample(300)


task = ClassificationTask(
    df,
    description="The dataset contains news articles categorized into four classes: World, Sports, Business, and Tech. The task is to classify each news article into one of the four categories.",
    x_column="text",
    y_column="label_text",
    eval_strategy="subsample",
    n_subsamples=20,
)

initial_prompts = [
    "What is the primary category of this news piece? Choose from World, Sports, Business, or Tech. Place your selected category between <final_answer> </final_answer>.",
    "Analyze this news article and categorize it as either World, Sports, Business, or Tech. Format your answer within <final_answer> </final_answer> tags.",
    "Your task is to identify whether this news article belongs to World, Sports, Business, or Tech news. Provide your classification between the markers <final_answer> </final_answer>.",
    "Please review the following news content and classify it into one of these categories: World, Sports, Business, or Tech. Your answer must be formatted with <final_answer> </final_answer> tags.",
    "Based on the content, determine if this news article falls under World, Sports, Business, or Tech category. Return only your classification within <final_answer> </final_answer>.",
    "Examine this news article and identify its primary category (World, Sports, Business, or Tech). Your final classification should be enclosed between <final_answer> </final_answer> markers.",
]


llm = APILLM(
    api_url=args.base_url,
    model_id=args.model_id,
    api_key=args.api_key,
)

downstream_llm = llm
meta_llm = llm

predictor = MarkerBasedClassifier(downstream_llm, classes=task.classes)

optimizer = EvoPromptGA(
    task=task,
    prompt_template=EVOPROMPT_GA_TEMPLATE,
    predictor=predictor,
    meta_llm=meta_llm,
    initial_prompts=initial_prompts,
    callbacks=callbacks,
)

best_prompts = optimizer.optimize(n_steps=args.n_steps)

logger.info(f"Optimized prompts: {best_prompts}")
