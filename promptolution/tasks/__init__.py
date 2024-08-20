from typing import List
import json
from pathlib import Path

from promptolution.tasks.base_task import DummyTask, BaseTask
from promptolution.tasks.classification_tasks import ClassificationTask

def get_tasks(config) -> List[BaseTask]:
    task_names = config.task_name.split(",")
    task_descriptions_path = Path(config.task_descriptions_path)
    task_descriptions = json.loads(task_descriptions_path.read_text())

    task_list = []
    for task_name in task_names:
        if task_name == "dummy":
            task = DummyTask()
            task_list.append(task)
            continue
        task = ClassificationTask(task_name, task_descriptions[task_name])
        task_list.append(task)

    return task_list