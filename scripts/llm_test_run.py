"""Test script for measuring raw LLM inference performance on a dataset."""
import time
from tqdm import tqdm
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--datasets", type=list, default=["agnews", "subj"])
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-storage-path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    if "vllm" in args.model:
        llm = get_llm(
            args.model,
            batch_size=args.batch_size,
            model_storage_path=args.model_storage_path,
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
        predictor = Classificator(llm, classes=task.classes)

        prompt = task.initial_population

        xs = task.xs[:config.n_eval_samples]
        ys = task.ys[:config.n_eval_samples]

        for prompt in tqdm(task.initial_population):
            preds, seqs = predictor.predict(prompt, xs, return_seq=True)

            scores = []
            for i in range(len(xs)):
                scores.append(1 if preds[0][i] == ys[i] else 0)

            # clean up the sequences
            seqs = [seq.replace("\n", "").strip() for seq in seqs]

            # if single prompts should be stored
            # df = pd.DataFrame(dict(prompt=prompt, seq=seqs, score=scores))
            # df.to_csv(args.output + "_detailed", index=False)

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
            results.to_csv(args.output, mode="a", header=False, index=False)

    total_inference_time = time.time() - start_time
    print(f"Total inference took {total_inference_time:.2f} seconds")


if __name__ == "__main__":
    main()
