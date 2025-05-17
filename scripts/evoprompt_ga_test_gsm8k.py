"""Test run for the Opro optimizer."""

import argparse
import random
from logging import Logger
from datasets import load_dataset

from promptolution import (
    LoggerCallback,
    TokenCountCallback,
    FileOutputCallback,
    EVOPROMPT_GA_TEMPLATE,
    get_llm,
    ClassificationTask,
    MarkerBasedClassifier,
    EvoPromptGA
)


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

df = load_dataset("openai/gsm8k", name="main", split="train", revision="main").to_pandas().sample(300, random_state=args.seed)

df["input"] = df["question"]
df["target"] = df["answer"].str.extract(r"#### (.*)")

task = ClassificationTask(
    df,
    description="The dataset consists of elementary school math word problems that require multi-step reasoning to solve. The task is to solve each word problem and provide the final answer.",
    x_column="input",
    y_column="target",
)

initial_prompts = [
    "Solve this math problem and put your answer between <final_answer> and </final_answer> tags.",
    "Read the following grade school word problem carefully. Show your reasoning and provide the final answer within <final_answer> tags.",
    "What's the answer? Remember to format it as <final_answer>your answer</final_answer>.",
    "I need help with this math problem. Please solve it step-by-step and clearly mark your final answer using <final_answer> </final_answer>.",
    "These word problems require multi-step reasoning. Work through the problem methodically, then place your numerical answer between <final_answer> </final_answer> markers.",
    "Solve the problem. Answer format: <final_answer>answer</final_answer>",
    "You are a math tutor helping elementary students with word problems. Explain your reasoning clearly, then provide your answer in the format <final_answer>answer</final_answer>.",
    "Examine this multi-step math problem. Calculate the solution and ensure you format your final answer within <final_answer> tags as instructed.",
    "Find the solution to this word problem using logical reasoning. Your final response must be formatted as <final_answer>your calculated result</final_answer>.",
    "Basic arithmetic word problem below. Solve it and clearly indicate your answer between <final_answer> and </final_answer> markers for easy evaluation.",
    "Could you please solve this math word problem? I need the final answer wrapped in <final_answer> tags.",
    "Analyze and solve the following elementary school math word problem that requires multiple steps of reasoning. Your answer should be provided in this exact format: <final_answer>your numerical answer</final_answer>.",
    "Math problem. Solve. Put answer in <final_answer></final_answer>.",
    "Below is a grade school mathematics word problem that may require multiple steps to solve. Please work through it carefully and make sure to format your final numerical answer as <final_answer>answer</final_answer>.",
    "Kindly solve this word problem by applying appropriate mathematical operations. Remember that your final answer must be enclosed within <final_answer> </final_answer> tags for proper evaluation.",
    "This dataset contains elementary math word problems. Read carefully, solve step by step, and format your answer between <final_answer> </final_answer> tags.",
    "I'm practicing math word problems that require multi-step reasoning. Help me solve this one and put the answer in <final_answer>answer</final_answer> format.",
    "Solve the following arithmetic word problem. The answer should be a number placed between <final_answer> and </final_answer> tags. No explanations needed - just the formatted answer.",
    "You're given a mathematical word problem from elementary school. Your task is to solve it using logical reasoning and mathematical operations. Present your final answer using this format: <final_answer>your answer</final_answer>.",
    "Word problem ahead! Use your math skills to find the answer, then format it exactly like this: <final_answer>your numerical solution</final_answer>.",
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

predictor = MarkerBasedClassifier(downstream_llm, classes=None)

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
