"""Module for task-related functions and classes."""
import pandas as pd

from promptolution.config import ExperimentConfig
from promptolution.tasks.base_task import BaseTask
from promptolution.tasks.classification_tasks import ClassificationTask


def get_task(df: pd.DataFrame, config: ExperimentConfig) -> BaseTask:
    """Get the task based on the provided DataFrame and configuration.

    So far only ClassificationTask is supported.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        config (ExperimentConfig): Configuration for the experiment.

    Returns:
        BaseTask: An instance of a task class based on the provided DataFrame and configuration.
    """
    return ClassificationTask(df, config=config)
