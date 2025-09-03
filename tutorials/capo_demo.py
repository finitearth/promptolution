"""Test run for the Opro optimizer."""


import argparse
from logging import Logger

from datasets import load_dataset

from promptolution.llms import APILLM
from promptolution.optimizers import CAPO
from promptolution.predictors import MarkerBasedClassifier
from promptolution.tasks import ClassificationTask
from promptolution.utils import FileOutputCallback, LoggerCallback, TokenCountCallback

logger = Logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="results/capo/")
parser.add_argument("--api_url", default="https://api.openai.com/v1")
parser.add_argument("--model_id", default="gpt-4-0613")
parser.add_argument("--n-steps", type=int, default=2)
parser.add_argument("--api_key", default=None)
args = parser.parse_args()

callbacks = [
    LoggerCallback(logger),
    FileOutputCallback(args.output_dir, file_type="csv"),
    TokenCountCallback(100000, "input_tokens"),
]

df = load_dataset("openai/gsm8k", name="main", split="train", revision="main").to_pandas().sample(400)

df["input"] = df["question"]
df["target"] = df["answer"].str.extract(r"#### (.*)")

task = ClassificationTask(
    df,
    task_description="The dataset consists of elementary school math word problems that require multi-step reasoning to solve. The task is to solve each word problem and provide the final answer.",
    x_column="input",
    y_column="target",
    eval_strategy="sequential_block",
)

initial_prompts = [
    "Solve this math problem and put your answer between <final_answer> and </final_answer> tags.",
    "Read the following grade school word problem carefully. Show your reasoning and provide the final answer within <final_answer> tags.",
    "What's the answer? Remember to format it as <final_answer>your answer</final_answer>.",
    "I need help with this math problem. Please solve it step-by-step and clearly mark your final answer using <final_answer> </final_answer>.",
    "These word problems require multi-step reasoning. Work through the problem methodically, then place your numerical answer between <final_answer> </final_answer> markers.",
    "Solve the problem. Answer format: <final_answer>answer</final_answer>",
    "You are a math tutor helping elementary students with word problems. Explain your reasoning clearly, then provide your answer in the format <final_answer>answer</final_answer>.",
]

llm = APILLM(model_id=args.model_id, api_key=args.api_key, api_url=args.api_url)

downstream_llm = llm
meta_llm = llm

predictor = MarkerBasedClassifier(downstream_llm, classes=None)

optimizer = CAPO(
    task=task,
    predictor=predictor,
    meta_llm=meta_llm,
    initial_prompts=initial_prompts,
    callbacks=callbacks,
)

best_prompts = optimizer.optimize(n_steps=args.n_steps)

logger.info(f"Optimized prompts: {best_prompts}")
