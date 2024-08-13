from promptolution.tasks import Task
from typing import List

class Optimizer:
    def __init__(self, intial_prompts: List[str], task: Task):
        pass

    def step(self) -> List[str]:
        pass

class DummyOptimizer(Optimizer):
    def __init__(self, initial_prompts: List[str], task: Task):
        self.prompts = initial_prompts
        self.task = task

    def step(self) -> List[str]:
        return self.prompts
    

class EvoPromptDE(Optimizer):
    def __init__(self, initial_prompts: List[str], task: Task):
        self.prompts = initial_prompts
        self.task = task

    def step(self) -> List[str]:
        pass


class EvoPrompDEtWithTaskDesc(Optimizer):
    def __init__(self, initial_prompts, task, task_desc):
        self.prompts = initial_prompts
        self.task = task
        self.task_desc = task_desc

    def step(self) -> List[str]:
        pass


class EvoPromptEA(Optimizer):
    def __init__(self, initial_prompts, task):
        self.prompts = initial_prompts
        self.task = task

    def step(self) -> List[str]:
        pass


class EvoPrompEAtWithTaskDesc(Optimizer):
    def __init__(self, initial_prompts, task, task_desc):
        self.prompts = initial_prompts
        self.task = task
        self.task_desc = task_desc

    def step(self) -> List[str]:
        pass