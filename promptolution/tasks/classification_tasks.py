import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask

class ClassificationTask(BaseTask):
    def __init__(self, task_id: str, dataset_json: Dict, seed: int = 42):
        self.task_id: str = task_id
        self.dataset_json: Dict = dataset_json
        self.description: Optional[str] = None
        self.initial_population: Optional[List[str]] = None
        self.xs: Optional[np.ndarray] = np.array([])
        self.ys: Optional[np.ndarray] = None
        self.classes: Optional[List] = None
        self._parse_task()
        self.reset_seed(seed)


    def __str__(self):
        return self.task_id

    def _parse_task(self):
        task_path = Path(self.dataset_json["path"])
        self.description = self.dataset_json["description"] #TODO move descriptions to their respective task_dir
        self.classes = self.dataset_json["classes"]

        with open(task_path / Path(self.dataset_json["init_prompts"]), "r", encoding="utf-8") as file:
            lines = file.readlines()
        self.initial_population = [line.strip() for line in lines]

        seed = Path(self.dataset_json["seed"])
        split = Path(self.dataset_json["split"] + ".txt")

        with open(task_path / seed / split, "r", encoding="utf-8") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]

        xs = []
        ys = []
        verbalizers = get_dataset_verbalizers(self.task_id)

        for line in lines:
            x, y = line.split("\t")
            xs.append(x)
            ys.append(verbalizers[int(y)])

        self.xs = np.array(xs)
        self.ys = np.array(ys)

    def evaluate(self, prompts: List[str], predictor: BasePredictor, n_samples: int = 20) -> np.ndarray: # nsamples -> 200 #TODO include in config
        if isinstance(prompts, str):
            prompts = [prompts]
        # Randomly select a subsample of n_samples
        indices = np.random.choice(len(self.xs), n_samples, replace=False)
        xs_subsample = self.xs[indices]
        ys_subsample = self.ys[indices]

        # Make predictions on the subsample
        preds = predictor.predict(prompts, xs_subsample)
        
        # Calculate accuracy: number of correct predictions / total number of predictions per prompt
        return np.mean(preds == ys_subsample, axis=1)
    
    def reset_seed(self, seed: int = None):
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)


def get_dataset_verbalizers(dataset: str) -> List[str]: #TODO move to task descriptions
    if dataset in ["sst2", "mr", "cr"]:
        verbalizers = ["negative", "positive"]
    elif dataset == "agnews":
        verbalizers = ["World", "Sports", "Business", "Tech"]
    elif dataset == "sst-5":
        verbalizers = [
            "terrible",
            "bad",
            "okay",
            "good",
            "great",
        ]
    elif dataset == "subj":
        verbalizers = ["subjective", "objective"]
    elif dataset == "trec":
        verbalizers = [
            "Description",
            "Entity",
            "Expression",
            "Human",
            "Location",
            "Number",
        ]
    else:
        raise ValueError(f"Dataset {dataset} not found.")

    return verbalizers
