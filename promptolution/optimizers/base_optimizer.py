from abc import ABC, abstractmethod
from typing import Callable, List

from promptolution.tasks.base_task import BaseTask


class BaseOptimizer(ABC):
    def __init__(self, initial_prompts: list[str], task: BaseTask, callbacks: list[Callable] = [], predictor=None):
        self.prompts = initial_prompts
        self.task = task
        self.callbacks = callbacks
        self.predictor = predictor

    @abstractmethod
    def optimize(self, n_steps: int) -> List[str]:
        raise NotImplementedError

    def _on_step_end(self):
        for callback in self.callbacks:
            callback.on_step_end(self)

    def _on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def _on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end(self)


class DummyOptimizer(BaseOptimizer):
    def __init__(self, initial_prompts, *args, **kwargs):
        self.callbacks = []
        self.prompts = initial_prompts

    def optimize(self, n_steps) -> list[str]:
        self._on_step_end()
        self._on_epoch_end()
        self._on_train_end()
        return self.prompts
