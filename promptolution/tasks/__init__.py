from typing import List
import json
from pathlib import Path

from promptolution.tasks.base_task import DummyTask, BaseTask
from promptolution.tasks.classification_tasks import ClassificationTask

def get_tasks(config) -> List[BaseTask]:
    task_names = config.task_name.split(",")

    task_list = []
    for task_name in task_names:
        task_description_path = Path(config.ds_path) / Path(task_name) / Path("description.json")
        task_description = json.loads(task_description_path.read_text())
        if task_name == "dummy":
            task = DummyTask()
            task_list.append(task)
            continue
        task = ClassificationTask(task_name, task_description, seed=config.random_seed)
        task_list.append(task)

    return task_list
