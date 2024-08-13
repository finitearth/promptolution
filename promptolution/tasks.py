from typing import List, Dict, Optional
from configparser import ConfigParser
from pathlib import Path
import json
import numpy as np

from promptolution.predictor import Predictor


class Task:
    def __init__(self, task_id: str, dataset_json: Dict):
        self.task_id: str = task_id
        self.dataset_json: Dict = dataset_json
        self.description: Optional[str] = None
        self.initial_population: Optional[List[str]] = None
        self.xs: Optional[np.ndarray] = np.array([])
        self.ys: Optional[np.ndarray] = None
        self.classes: Optional[List] = None
        self._parse_task()

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

    def evaluate(self, prompt: str, predictor: Predictor):
        preds = predictor.predict(prompt, self.xs)
        return np.mean(preds == self.ys)


class DummyTask(Task):
    def __init__(self):
        self.task_id = "dummy"
        self.dataset_json = None
        self.initial_population = ["Some", "initial", "prompts"]
        self.description = "This is a dummy task for testing purposes."
        self.xs = np.array(["This is a test", "This is another test", "This is a third test"])
        self.ys = np.array(["positive", "negative", "positive"])
        self.classes = ["negative", "positive"]

    def evaluate(self, prompt: str, predictor: Predictor):
        return np.random.rand()


def get_tasks(config: ConfigParser) -> List[Task]:
    task_names = config["tasks"]["task_names"].split(",")
    task_descriptions_path = Path(config.get("tasks", "task_descriptions_path"))
    task_descriptions = json.loads(task_descriptions_path.read_text())

    task_list = []
    for task_name in task_names:
        if task_name == "dummy":
            task = DummyTask()
            task_list.append(task)
            continue
        task = Task(task_name, task_descriptions[task_name])
        task_list.append(task)

    return task_list


def get_dataset_verbalizers(dataset: str) -> List[str]:
    if dataset in ["sst2", "mr", "cr"]:
        verbalizers = ["negative", "positive"]  # num_classes
    elif dataset == "agnews":
        verbalizers = ["World", "Sports", "Business", "Tech"]  # num_classes
    elif dataset == "sst-5":
        verbalizers = [
            "terrible",
            "bad",
            "okay",
            "good",
            "great",
        ]  # num_classes
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
