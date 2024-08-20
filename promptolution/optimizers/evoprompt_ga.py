from promptolution.optimizers.base_optimizer import BaseOptimizer
from typing import List
import numpy as np

# from promptolution.tasks.base_task import BaseTask


class EvoPromptGA(BaseOptimizer):
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
