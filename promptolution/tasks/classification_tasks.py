import json
from pathlib import Path
from typing import Dict, List, Optional, Literal

import numpy as np

from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask

class ClassificationTask(BaseTask):
    def __init__(self, task_id: str, dataset_json: Dict, seed: int = 42, split: Literal["dev", "test"] = "dev"):
        self.task_id: str = task_id
        self.dataset_json: Dict = dataset_json
        self.description: Optional[str] = None
        self.initial_population: Optional[List[str]] = None
        self.xs: Optional[np.ndarray] = np.array([])
        self.ys: Optional[np.ndarray] = None
        self.classes: Optional[List] = None
        self.split: Literal["dev", "test"] = split
        self._parse_task()
        self.reset_seed(seed)


    def __str__(self):
        return self.task_id

    def _parse_task(self):
        task_path = Path(self.dataset_json["path"])
        self.description = self.dataset_json["description"]
        self.classes = self.dataset_json["classes"]

        with open(task_path / Path(self.dataset_json["init_prompts"]), "r", encoding="utf-8") as file:
            lines = file.readlines()
        self.initial_population = [line.strip() for line in lines]

        seed = Path(self.dataset_json["seed"])
        split = Path(self.split + ".txt")

        with open(task_path / seed / split, "r", encoding="utf-8") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]

        xs = []
        ys = []

        for line in lines:
            x, y = line.split("\t")
            xs.append(x)
            ys.append(self.classes[int(y)])

        self.xs = np.array(xs)
        self.ys = np.array(ys)

    def evaluate(self, prompts: List[str], predictor: BasePredictor, n_samples: int = 20, subsample: bool = True) -> np.ndarray: #TODO include in config
        if isinstance(prompts, str):
            prompts = [prompts]
        # Randomly select a subsample of n_samples
        if subsample:
            indices = np.random.choice(len(self.xs), n_samples, replace=False)
        else:
            indices = np.arange(len(self.xs))
            
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
