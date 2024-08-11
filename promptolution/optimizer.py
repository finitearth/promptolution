from promptolution.tasks import Task


class Optimizer:
    def __init__(self, intial_prompts: list[str], task: Task):
        pass

    def step(self) -> str:
        pass


class EvoPromptDE(Optimizer):
    def __init__(self, initial_prompts: list[str], task: Task):
        self.prompts = initial_prompts
        self.task = task

    def step(self) -> str:
        pass


class EvoPrompDEtWithTaskDesc(Optimizer):
    def __init__(self, initial_prompts, task, task_desc):
        self.prompts = initial_prompts
        self.task = task
        self.task_desc = task_desc

    def step(self) -> str:
        pass


class EvoPromptEA(Optimizer):
    def __init__(self, initial_prompts, task):
        self.prompts = initial_prompts
        self.task = task

    def step(self) -> str:
        pass


class EvoPrompEAtWithTaskDesc(Optimizer):
    def __init__(self, initial_prompts, task, task_desc):
        self.prompts = initial_prompts
        self.task = task
        self.task_desc = task_desc

    def step(self) -> str:
        pass