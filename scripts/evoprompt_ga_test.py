"""Test run for the Opro optimizer."""

import argparse
import random
from logging import Logger

from promptolution.callbacks import LoggerCallback, FileOutputCallback, TokenCountCallback
from promptolution.templates import EVOPROMPT_GA_TEMPLATE
from promptolution.helpers import get_llm
from promptolution.tasks import ClassificationTask
from promptolution.predictors import MarkerBasedClassifier
from promptolution.optimizers import EvoPromptGA
from datasets import load_dataset

logger = Logger(__name__)

"""Run a test run for any of the implemented optimizers."""
parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--model-storage-path", default="../models/")
parser.add_argument("--output-dir", default="results/evoprompt_ga_test/")
parser.add_argument("--max-model-len", type=int, default=1024)
parser.add_argument("--n-steps", type=int, default=2)
parser.add_argument("--n-eval-samples", type=int, default=20)
parser.add_argument("--token", default=None)
parser.add_argument("--seed", type=int, default=187)
args = parser.parse_args()

callbacks = [
    LoggerCallback(logger),
    FileOutputCallback(args.output_dir, file_type="csv"),
    TokenCountCallback(100000, "input_tokens"),
]

df = load_dataset("SetFit/ag_news", split="train", revision="main").to_pandas().sample(300, random_state=args.seed)

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
    "What is the primary category of this news piece? Choose from World, Sports, Business, or Tech. Place your selected category between <final_answer> </final_answer>.",
    "Analyze this news article and categorize it as either World, Sports, Business, or Tech. Format your answer within <final_answer> </final_answer> tags.",
    "Your task is to identify whether this news article belongs to World, Sports, Business, or Tech news. Provide your classification between the markers <final_answer> </final_answer>.",
    "Please review the following news content and classify it into one of these categories: World, Sports, Business, or Tech. Your answer must be formatted with <final_answer> </final_answer> tags.",
    "Based on the content, determine if this news article falls under World, Sports, Business, or Tech category. Return only your classification within <final_answer> </final_answer>.",
    "Examine this news article and identify its primary category (World, Sports, Business, or Tech). Your final classification should be enclosed between <final_answer> </final_answer> markers.",
    "In this task, you must categorize a news article into one of four classes: World, Sports, Business, or Tech. Remember to place your answer between <final_answer> </final_answer> tags for proper evaluation.",
    "Read the provided news excerpt carefully and assign it to either World, Sports, Business, or Tech category. Ensure your answer appears between <final_answer> </final_answer> tags.",
    "Considering the main subject matter, classify this news article as World, Sports, Business, or Tech. Format your response with <final_answer> </final_answer>.",
    "Determine the appropriate category for this news article from the following options: World, Sports, Business, or Tech. Your selected category must be placed within <final_answer> </final_answer> markers.",
    "After analyzing the given news article, assign it to the most suitable category: World, Sports, Business, or Tech. Your classification should be enclosed in <final_answer> </final_answer> tags.",
    "Your objective is to classify the news article into one of the following categories: World, Sports, Business, or Tech based on its primary focus. Submit your answer between <final_answer> </final_answer> tags.",
    "Which category best describes this news article: World, Sports, Business, or Tech? Provide your answer inside <final_answer> </final_answer> markers.",
    "As a content classifier, determine if the following news article belongs to World, Sports, Business, or Tech news. Place your answer within <final_answer> </final_answer> tags.",
    "Evaluate the following news article and indicate whether it primarily concerns World, Sports, Business, or Tech topics. Your classification must appear between <final_answer> </final_answer>.",
    "Given a news article, your task is to determine its primary category from World, Sports, Business, or Tech. The final classification must be provided between <final_answer> </final_answer> tags.",
    "Conduct a thorough analysis of the provided news article and classify it as belonging to one of these four categories: World, Sports, Business, or Tech. Your answer should be presented within <final_answer> </final_answer> markers.",
    "Simply indicate whether this news article is about World, Sports, Business, or Tech. Include your answer between <final_answer> </final_answer> tags.",
]

# randomly sample 5 initial prompts
initial_prompts = random.sample(initial_prompts, 5)

if "vllm" in args.model:
    llm = get_llm(
        args.model,
        batch_size=None,
        max_model_len=args.max_model_len,
        model_storage_path=args.model_storage_path,
        revision="main",
    )
else:
    llm = get_llm(args.model, args.token)

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
    n_eval_samples=args.n_eval_samples,
)

best_prompts = optimizer.optimize(n_steps=args.n_steps)

logger.info(f"Optimized prompts: {best_prompts}")
