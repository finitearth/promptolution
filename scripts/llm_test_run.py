"""Test script for measuring raw LLM inference performance on a dataset."""
import argparse
import time
from logging import Logger
import os

import numpy as np
import pandas as pd
from promptolution.config import Config
from promptolution.llms import get_llm
from promptolution.predictors import FirstOccurrenceClassificator
from promptolution.tasks import get_task
from tqdm import tqdm

logger = Logger(__name__)

"""Run inference test on a dataset using a specified LLM."""
parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--output-dir", default="results/")
parser.add_argument("--datasets", default=["subj"])
parser.add_argument("--token", default=None)
parser.add_argument("--batch-size", default=None)
parser.add_argument("--revision", default="main")
parser.add_argument("--max-model-len", default=None)
parser.add_argument("--model-storage-path", default=None)
args = parser.parse_args()

# make sure the output directory exist
os.makedirs(args.output_dir, exist_ok=True)

start_time = time.time()

if args.max_model_len is not None:
    max_model_len = int(args.max_model_len)

if "vllm" in args.model:
    llm = get_llm(
        args.model,
        batch_size=args.batch_size,
        max_model_len=max_model_len,
        model_storage_path=args.model_storage_path,
        revision=args.revision,
    )
else:
    llm = get_llm(args.model, args.token)

results = pd.DataFrame()

for dataset in args.datasets:
    config = Config(
        evaluation_llm=args.model,
        ds_path=f"data_sets/cls/{dataset}/",
        task_name=dataset,
        api_token=args.token,
        n_eval_samples=200,
    )

    task = get_task(config, split="dev")
    predictor = FirstOccurrenceClassificator(llm, classes=task.classes)

    prompts = [task.initial_population[0]]

    xs = task.xs[: config.n_eval_samples]
    ys = task.ys[: config.n_eval_samples]

    for prompt in tqdm(prompts):
        preds, seqs = predictor.predict(prompt, xs, return_seq=True)

        scores = []
        for i in range(len(xs)):
            scores.append(1 if preds[0][i] == ys[i] else 0)

        # clean up the sequences
        seqs = [seq.replace("\n", "").strip() for seq in seqs]

        # if single prompts should be stored
        # df = pd.DataFrame(dict(prompt=prompt, seq=seqs, score=scores))
        # df.to_csv(args.output_dir + "results_detailed.csv", index=False)

        accuracy = np.array(scores).mean()

        results = pd.DataFrame(
            dict(
                model=args.model,
                dataset=dataset,
                prompt=prompt,
                accuracy=accuracy,
                n_samples=len(xs),
            ),
            index=[0],
        )

        if not os.path.exists(args.output_dir + "llm_test_results.csv"):
            results.to_csv(args.output_dir + "llm_test_results.csv", index=False)
        else:
            results.to_csv(args.output_dir + "llm_test_results.csv", mode="a", header=False, index=False)

total_inference_time = time.time() - start_time
print(
    f"Total inference took {total_inference_time:.2f} seconds and required {llm.get_token_count()} tokens."
)
print(f"Results saved to {args.output_dir}")
