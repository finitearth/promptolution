"""Module for task-related functions and classes."""
import json
from pathlib import Path
from typing import List, Literal

from promptolution.tasks.base_task import BaseTask, DummyTask
from promptolution.tasks.classification_tasks import ClassificationTask


def get_task(
    ds_path: str, split: Literal["dev", "test"] = "dev", random_seed: int = 42, task_name: str = "Classification Task"
) -> BaseTask:
    """Create and return the task instance based on the provided configuration.

    This function supports creating multiple tasks, including a special 'dummy' task
    for testing purposes and classification tasks based on JSON descriptions.

    Args:
        config: Configuration object containing task settings.
                Expected attributes:
                - task_name (str): Comma-separated list of task names.
                - ds_path (str): Path to the dataset directory.
                - random_seed (int): Seed for random number generation.
        split (Literal["dev", "test"], optional): Dataset split to use. Defaults to "dev".

    Returns:
        BaseTask: A list of instantiated task objects.

    Raises:
        FileNotFoundError: If the task description file is not found.
        json.JSONDecodeError: If the task description file is not valid JSON.

    Notes:
        - The 'dummy' task is a special case that creates a DummyTask instance.
        - For all other tasks, a ClassificationTask instance is created.
        - The task description is loaded from a 'description.json' file in the dataset path.
    """
    task_description_path = Path(ds_path) / Path("description.json")
    task_description = json.loads(task_description_path.read_text())
    if task_name == "dummy":
        task = DummyTask()
        return task
    task = ClassificationTask(task_name, task_description, split=split, seed=random_seed)

    return task
