from typing import Callable, List

import numpy as np

from promptolution.tasks import Task


def get_optimizer(name, **args):
    if name == "dummy":
        return DummyOptimizer(**args)
    if name == "evopromptde":
        return EvoPromptDE(**args)
    if name == "evopromptga":
        return EvoPromptGA(**args)
    raise ValueError(f"Unknown optimizer: {name}")


class Optimizer:
    def __init__(self, initial_prompts: list[str], task: Task, callbacks: list[Callable] = [], predictor=None):
        self.prompts = initial_prompts
        self.task = task
        self.callbacks = callbacks
        self.predictor = predictor

    def step(self) -> str:
        raise NotImplementedError

    def _on_step_end(self):
        for callback in self.callbacks:
            callback.on_step_end(self)


class DummyOptimizer(Optimizer):
    def __init__(self, initial_prompts: list[str], task: Task, callbacks: list[Callable] = []):
        super().__init__(initial_prompts, task, callbacks)

    def step(self) -> list[str]:
        self._on_step_end()
        return self.prompts

    def _on_step_end(self):
        for callback in self.callbacks:
            callback.on_step_end(self)


class EvoPromptGA(Optimizer):
    def __init__(self, prompt_template, meta_llm, selection_mode="wheel", **args):
        self.prompt_template = prompt_template
        self.meta_llm = meta_llm
        assert selection_mode in ["random", "wheel", "tour"], "Invalid selection mode."
        self.selection_mode = selection_mode
        super().__init__(**args)

    def step(self) -> str:
        # get scores from task
        self.scores = [self.task.evaluate(prompt, self.predictor) for prompt in self.prompts]

        # sort prompts by score
        self.prompts = [prompt for _, prompt in sorted(zip(self.scores, self.prompts), reverse=True)]

        if self.selection_mode == "wheel":
            wheel_idx = np.random.choice(
                np.arange(0, len(self.prompts)),
                size=len(self.prompts),
                replace=True,
                p=np.array(self.scores) / np.sum(self.scores),
            ).tolist()
            parent_pop = [self.prompts[idx] for idx in wheel_idx]

        elif self.selection_mode in ["random", "tour"]:
            parent_pop = self.prompts

        # crossover
        for i in range(len(self.prompts)):
            print("Crossover", i)
            new_pop = []

            if self.selection_mode in ["random", "wheel"]:
                parent_1, parent_2 = np.random.choice(parent_pop, size=2, replace=False)
            elif self.selection_mode == "tour":
                group_1 = np.random.choice(parent_pop, size=2, replace=False)
                group_2 = np.random.choice(parent_pop, size=2, replace=False)
                parent_1 = group_1[np.argmax([self.task.evaluate(prompt, self.predictor) for prompt in group_1])]
                parent_2 = group_2[np.argmax([self.task.evaluate(prompt, self.predictor) for prompt in group_2])]

            meta_prompt = self.prompt_template.replace("<prompt1>", parent_1).replace("<prompt2>", parent_2)

            child_prompt = self.meta_llm.get_response(meta_prompt)
            child_prompt = child_prompt.split("<prompt>")[-1].split("</prompt>")[0].strip()

            new_pop.append(child_prompt)

        # eliminate worst
        prompts = self.prompts + new_pop
        scores = self.scores + [self.task.evaluate(prompt, self.predictor) for prompt in new_pop]
        self.prompts = [prompt for _, prompt in sorted(zip(scores, prompts), reverse=True)][: len(self.prompts)]

        self._on_step_end()
        return self.prompts


class EvoPromptDE(Optimizer):
    def __init__(self, prompt_template, meta_llm, donor_random=False, **args):
        self.prompt_template = prompt_template
        self.donor_random = donor_random
        self.meta_llm = meta_llm
        super().__init__(**args)

    def step(self) -> List:
        self.scores = [self.task.evaluate(prompt, self.predictor) for prompt in self.prompts]
        self.prompts = [prompt for _, prompt in sorted(zip(self.scores, self.prompts), reverse=True)]
        cur_best = self.prompts[0]

        for i in range(len(self.prompts)):
            old_prompt = self.prompts[i]
            old_score = self.scores[i]

            canidates = [prompt for prompt in self.prompts if prompt != old_prompt]
            a, b, c = np.random.choice(canidates, size=3, replace=False)

            if not self.donor_random:
                c = cur_best

            meta_prompt = (
                self.prompt_template.replace("<prompt0>", old_prompt)
                .replace("<prompt1>", a)
                .replace("<prompt2>", b)
                .replace("<prompt3>", c)
            )

            child_prompt = self.meta_llm.get_response(meta_prompt)
            child_prompt = child_prompt.split("<prompt>")[-1].split("</prompt>")[0].strip()

            child_score = self.task.evaluate(child_prompt, self.predictor)

            if child_score > old_score:
                self.prompts[i] = child_prompt
                self.scores[i] = child_score

        self._on_step_end()
        return self.prompts
