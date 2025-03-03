"""Test script for measuring raw LLM inference performance on a dataset."""
import time
import json
from logging import Logger
import argparse
import pandas as pd
import numpy as np

from promptolution.tasks import get_task
from promptolution.config import Config
from promptolution.predictors import Classificator
from promptolution.llms import get_llm

logger = Logger(__name__)


def main():
    """Run inference test on a dataset using a specified LLM."""
    parser = argparse.ArgumentParser(description="Test LLM inference performance")
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--dataset", type=str, default="agnews")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--model-storage-path", type=str, default=None)
    args = parser.parse_args()

    config = Config(
        evaluation_llm=args.model,
        ds_path=f"data_sets/cls/{args.dataset}/",
        task_name=args.dataset,
        api_token=args.token,
        n_eval_samples=200,
    )

    start_time = time.time()

    task = get_task(config, split="dev")
    if "vllm" in args.model:
        llm = get_llm(
            config.evaluation_llm,
            model_storage_path=args.model_storage_path,
        )
    else:
        llm = get_llm(config.evaluation_llm, token=config.api_token)

    predictor = Classificator(llm, classes=task.classes)

    prompt = task.initial_population[0]

    xs = task.xs[:config.n_eval_samples]
    ys = task.ys[:config.n_eval_samples]

    preds, seqs = predictor.predict(prompt, xs, return_seq=True)

    scores = []
    for i in range(len(xs)):
        scores.append(1 if preds[0][i] == ys[i] else 0)

    # clean up the sequences
    seqs = [seq.replace("\n", "").strip() for seq in seqs]

    df = pd.DataFrame(dict(prompt=task.initial_population[0], seq=seqs, score=scores))

    total_inference_time = time.time() - start_time

    accuracy = np.array(scores).mean()

    print(f"Overall Acc {accuracy:.4f}")
    print(f"Used model {args.model} on dataset {args.dataset}")
    print(f"Total inference took {total_inference_time:.2f} seconds")

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
