from typing import Callable, List

import numpy as np
from tqdm import tqdm

from promptolution.tasks import Task


def get_optimizer(config, *args, **kwargs):
    if config.optimizer == "dummy":
        return DummyOptimizer(*args, **kwargs)
    if config.optimizer == "evopromptde":
        return EvoPromptDE(donor_random=config.donor_random, *args, **kwargs)
    if config.optimizer == "evopromptga":
        return EvoPromptGA(selection_mode=config.selection_mode, *args, **kwargs)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


class Optimizer:
    def __init__(self, initial_prompts: list[str], task: Task, callbacks: list[Callable] = [], predictor=None):
        self.prompts = initial_prompts
        self.task = task
        self.callbacks = callbacks
        self.predictor = predictor

    def optimize(self) -> str:
        raise NotImplementedError

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

    def optimize(self, n_steps: int) -> List[str]:
        # get scores from task
        self.scores = self.task.evaluate(self.prompts, self.predictor).tolist()
        # sort prompts by score
        self.prompts = [prompt for _, prompt in sorted(zip(self.scores, self.prompts), reverse=True)]
        self.scores = sorted(self.scores, reverse=True)

        for _ in range(n_steps):
            new_prompts = self._crossover(self.prompts, self.scores)
            prompts = self.prompts + new_prompts
            scores = self.scores + self.task.evaluate(new_prompts, self.predictor).tolist()

            # sort scores and prompts
            self.prompts = [prompt for _, prompt in sorted(zip(scores, prompts), reverse=True)][:len(self.prompts)]
            self.scores = sorted(scores, reverse=True)[:len(self.prompts)]

            self._on_step_end()
        return self.prompts

    def _crossover(self, prompts, scores) -> str:
        # parent selection
        if self.selection_mode == "wheel":
            wheel_idx = np.random.choice(
                np.arange(0, len(prompts)),
                size=len(prompts),
                replace=True,
                p=np.array(scores) / np.sum(scores) if np.sum(scores) > 0 else np.ones(len(scores)) / len(scores),
            ).tolist()
            parent_pop = [self.prompts[idx] for idx in wheel_idx]

        elif self.selection_mode in ["random", "tour"]:
            parent_pop = self.prompts

        # crossover
        meta_prompts = []
        for _ in self.prompts:
            if self.selection_mode in ["random", "wheel"]:
                parent_1, parent_2 = np.random.choice(parent_pop, size=2, replace=False)
            elif self.selection_mode == "tour":
                group_1 = np.random.choice(parent_pop, size=2, replace=False)
                group_2 = np.random.choice(parent_pop, size=2, replace=False)
                # use the best of each group based on scores
                parent_1 = group_1[np.argmax([self.scores[self.prompts.index(p)] for p in group_1])]
                parent_2 = group_2[np.argmax([self.scores[self.prompts.index(p)] for p in group_2])]

            meta_prompt = self.prompt_template.replace("<prompt1>", parent_1).replace("<prompt2>", parent_2)
            meta_prompts.append(meta_prompt)

        child_prompts = self.meta_llm.get_response(meta_prompts)
        child_prompts = [
            prompt.split("<prompt>")[-1].split("</prompt>")[0].strip() 
            for prompt in child_prompts
        ]

        return child_prompts


class EvoPromptDE(Optimizer):
    def __init__(self, prompt_template, meta_llm, donor_random=False, **args):
        self.prompt_template = prompt_template
        self.donor_random = donor_random
        self.meta_llm = meta_llm
        super().__init__(**args)

    def optimize(self, n_steps: int) -> List:
        self.scores = self.task.evaluate(self.prompts, self.predictor)
        self.prompts = [prompt for _, prompt in sorted(zip(self.scores, self.prompts), reverse=True)]
        self.scores = sorted(self.scores, reverse=True)

        for _ in range(n_steps):
            cur_best = self.prompts[0]
            meta_prompts = []
            for i in range(len(self.prompts)):
                # create meta prompts
                old_prompt = self.prompts[i]

                candidates = [prompt for prompt in self.prompts if prompt != old_prompt] 
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                if not self.donor_random:
                    c = cur_best

                meta_prompt = (
                    self.prompt_template.replace("<prompt0>", old_prompt)
                    .replace("<prompt1>", a)
                    .replace("<prompt2>", b)
                    .replace("<prompt3>", c)
                )

                meta_prompts.append(meta_prompt)

            child_prompts = self.meta_llm.get_response(meta_prompts)
            child_prompts = [
                prompt.split("<prompt>")[-1].split("</prompt>")[0].strip() 
                for prompt in child_prompts
            ]

            child_scores = self.task.evaluate(child_prompts, self.predictor)

            for i in range(len(self.prompts)):
                if child_scores[i] > self.scores[i]:
                    self.prompts[i] = child_prompts[i]
                    self.scores[i] = child_scores[i]

            self._on_step_end()

        return self.prompts


class DummyOptimizer(Optimizer):
    def __init__(self, initial_prompts, *args, **kwargs):
        self.callbacks = []
        self.prompts = initial_prompts

    def optimize(self, n_steps) -> list[str]:
        self._on_step_end()
        return self.prompts

            