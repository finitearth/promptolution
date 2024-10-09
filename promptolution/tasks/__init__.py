"""Module for task-related functions and classes."""
import json
from pathlib import Path
from typing import List, Literal

from promptolution.tasks.base_task import BaseTask, DummyTask
from promptolution.tasks.classification_tasks import ClassificationTask


def get_task(config, split: Literal["dev", "test"] = "dev") -> BaseTask:
    """Create and return the task instance based on the provided configuration.

    This function supports creating multiple tasks, including a special 'dummy' task
    for testing purposes and classification tasks based on JSON descriptions.

    Args:
        ds_path (str): Path to the dataset directory.
        random_seed (int): Seed for random number generation.
        split (Literal["dev", "test"], optional): Dataset split to use. Defaults to "dev".
        task_name (str): Comma-separated list of task names.

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
    if config.task_name == "dummy":
        task = DummyTask()
        return task
    task_description_path = Path(config.ds_path)
    task = ClassificationTask(task_description_path, config.task_name, split=split, seed=config.random_seed)

    return task
