"""Module for classification tasks."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

import numpy as np
from sklearn.metrics import accuracy_score

from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask


class ClassificationTask(BaseTask):
    """A class representing a classification task in the promptolution library.

    This class handles the loading and management of classification datasets,
    as well as the evaluation of predictors on these datasets.

    Attributes:
        task_id (str): Unique identifier for the task.
        path (Path): Path to the dataset description JSON file, and initial prompts.
        dataset_json (Dict): Dictionary containing dataset information.
        description (Optional[str]): Description of the task.
        initial_population (Optional[List[str]]): Initial set of prompts.
        xs (Optional[np.ndarray]): Input data for the task.
        ys (Optional[np.ndarray]): Ground truth labels for the task.
        classes (Optional[List]): List of possible class labels.
        seed (int): Random seed for reproducibility.
        split (Literal["dev", "test"]): Dataset split to use.
        metric (Callable): Metric to use as an evaluation score for the prompts.

    Inherits from:
        BaseTask: The base class for tasks in the promptolution library.
    """

    def __init__(
        self,
        dataset_path: Path,
        task_id: str = "Classification Task",
        seed: int = 42,
        split: Literal["dev", "test"] = "dev",
        metric: Callable = accuracy_score,
    ):
        """Initialize the ClassificationTask.

        Args:
            task_id (str): Unique identifier for the task.
            dataset_path (str): Path to the dataset description JSON file.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            split (Literal["dev", "test"], optional): Dataset split to use. Defaults to "dev".
            metric (Callable): Metric to use as an evaluation score for the prompts. Defaults to sklearn's accuracy.
        """
        self.task_id: str = task_id
        self.path: Path = dataset_path
        self.dataset_json: Dict = json.loads((dataset_path / Path("description.json")).read_text())
        self.description: Optional[str] = None
        self.initial_population: Optional[List[str]] = None
        self.xs: Optional[np.ndarray] = np.array([])
        self.ys: Optional[np.ndarray] = None
        self.classes: Optional[List] = None
        self.split: Literal["dev", "test"] = split
        self.metric = metric
        self._parse_task()
        self.reset_seed(seed)

    def __str__(self):
        """Convert task to string representation, returning the task id."""
        return self.task_id

    def _parse_task(self):
        """Parse the task data from the provided dataset JSON.

        This method loads the task description, classes, initial prompts,
        and the dataset split (dev or test) into the class attributes.
        """
        self.description = self.dataset_json["description"]
        self.classes = self.dataset_json["classes"]

        with open(self.path / Path(self.dataset_json["init_prompts"]), "r", encoding="utf-8") as file:
            lines = file.readlines()
        self.initial_population = [line.strip() for line in lines]

        seed = Path(self.dataset_json["seed"])
        split = Path(self.split + ".txt")

        with open(self.path / seed / split, "r", encoding="utf-8") as file:
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

    def evaluate(
        self,
        prompts: List[str],
        predictor: BasePredictor,
        n_samples: int = 20,
        subsample: bool = True,
        return_seq: bool = False,
    ) -> np.ndarray:
        """Evaluate a set of prompts using a given predictor.

        Args:
            prompts (List[str]): List of prompts to evaluate.
            predictor (BasePredictor): Predictor to use for evaluation.
            n_samples (int, optional): Number of samples to use if subsampling. Defaults to 20.
            subsample (bool, optional): Whether to use subsampling. Defaults to True.
            return_seq (bool, optional): rather to return the generating sequence

        Returns:
            np.ndarray: Array of accuracy scores for each prompt.
        """
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
        preds = predictor.predict(prompts, xs_subsample, return_seq=return_seq)

        if return_seq:
            preds, seqs = preds

        scores = np.array([self.metric(ys_subsample, pred) for pred in preds])

        if return_seq:
            return scores, seqs

        return scores

    def reset_seed(self, seed: int = None):
        """Reset the random seed."""
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)
