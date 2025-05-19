"""Test run for the Opro optimizer."""

import argparse
from logging import Logger

from datasets import load_dataset

from promptolution.llms import VLLM
from promptolution.optimizers import OPRO, OPRO_TEMPLATE_TD
from promptolution.predictors import MarkerBasedClassifier
from promptolution.tasks import ClassificationTask
from promptolution.utils import FileOutputCallback, LoggerCallback, TokenCountCallback

logger = Logger(__name__)

"""Run a test run for any of the implemented optimizers."""
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--model-storage-path", default="../models/")
parser.add_argument("--output-dir", default="results/opro_test/")
parser.add_argument("--max-model-len", type=int, default=2048)
parser.add_argument("--n-steps", type=int, default=999)
parser.add_argument("--seed", type=int, default=187)
args = parser.parse_args()

callbacks = [
    LoggerCallback(logger),
    FileOutputCallback(args.output_dir, file_type="csv"),
    TokenCountCallback(5000000, "input_tokens"),
]

df = load_dataset("SetFit/ag_news", split="train", revision="main").to_pandas().sample(300, random_state=args.seed)

task = ClassificationTask(
    df,
    description="The dataset contains news articles categorized into four classes: World, Sports, Business, and Tech. The task is to classify each news article into one of the four categories.",
    x_column="text",
    y_column="label_text",
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
]


llm = VLLM(
    model_id=args.model, model_storage_path=args.model_storage_path, max_model_len=args.max_model_len, seed=args.seed
)

downstream_llm = llm
meta_llm = llm

predictor = MarkerBasedClassifier(downstream_llm, classes=task.classes)

optimizer = OPRO(
    task=task,
    prompt_template=OPRO_TEMPLATE_TD.replace("<task_desc", task.description),
    predictor=predictor,
    meta_llm=meta_llm,
    initial_prompts=initial_prompts,
    callbacks=callbacks,
    max_num_instructions=20,
    num_instructions_per_step=8,
    num_few_shots=3,
)

best_prompts = optimizer.optimize(n_steps=args.n_steps)

logger.info(f"Optimized prompts: {best_prompts}")
