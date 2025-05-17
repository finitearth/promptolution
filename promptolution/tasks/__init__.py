"""Module for task-related functions and classes."""

# Re-export the factory function from helpers
from ..helpers import get_task
from .base_task import BaseTask
from .classification_tasks import ClassificationTask

# Define what symbols are exported by default when using 'from promptolution.tasks import *'
__all__ = ["BaseTask", "ClassificationTask", "get_task"]
