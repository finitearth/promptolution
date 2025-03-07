"""Module for task-related functions and classes."""
import json
from pathlib import Path
from typing import List, Literal, Union

import pandas as pd

from promptolution.tasks.base_task import BaseTask, DummyTask
from promptolution.tasks.classification_tasks import ClassificationTask


def get_task(
    config=None,
    split: Literal["dev", "test"] = "dev",
    dataset: Union[Path, pd.DataFrame] = None,
    task_name: str = None,
    random_seed: int = None,
) -> BaseTask:
    """Create and return an task instance.

    This function supports creating multiple tasks, including a special 'dummy' task
    for testing purposes and classification tasks based on parsed config, or alternativly
    the parsed arguments.

    Args:
        config (Config): Configuration object containing the task details.
        split (str): Split of the dataset to use for the task (default: 'dev').
        dataset (Path or pd.DataFrame): Path to the dataset containing the task description
            or a pandas DataFrame with the dataset.
        task_name (str): Name of the task to create.
        random_seed (int): Random seed for the task.

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
    if dataset is None:
        dataset = config.dataset
    if task_name is None:
        task_name = config.task_name
    if random_seed is None:
        random_seed = config.random_seed
    if task_name == "dummy":
        task = DummyTask()
        return task

    if isinstance(dataset, pd.DataFrame):
        initial_prompts = config.init_prompts
        task = ClassificationTask.from_dataframe(
            dataset, task_name, split=split, seed=random_seed, initial_prompts=initial_prompts
        )
    else:
        task_description_path = Path(dataset)
        task = ClassificationTask(task_description_path, task_name, split=split, seed=random_seed)

    return task
