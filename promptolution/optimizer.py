class Optimizer:
    def __init__(self):
        pass

    def step(self) -> str:
        pass


class EvoPrompt(Optimizer):
    def __init__(self, initial_prompts, task):
        self.prompts = initial_prompts
        self.task = task

    def step(self) -> str:
        pass


class EvoPromptWithTaskDesc(Optimizer):
    def __init__(self, initial_prompts, task, task_desc):
        self.prompts = initial_prompts
        self.task = task
        self.task_desc = task_desc

    def step(self) -> str:
        pass