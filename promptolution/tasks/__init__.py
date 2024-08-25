import json
from pathlib import Path
from typing import List, Literal

from promptolution.tasks.base_task import BaseTask, DummyTask
from promptolution.tasks.classification_tasks import ClassificationTask


def get_tasks(config, split: Literal["dev", "test"] = "dev") -> List[BaseTask]:
    task_names = config.task_name.split(",")

    task_list = []
    for task_name in task_names:
        task_description_path = Path(config.ds_path) / Path("description.json")
        task_description = json.loads(task_description_path.read_text())
        if task_name == "dummy":
            task = DummyTask()
            task_list.append(task)
            continue
        task = ClassificationTask(task_name, task_description, split=split, seed=config.random_seed)
        task_list.append(task)

    return task_list
