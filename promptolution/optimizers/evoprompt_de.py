import numpy as np
from typing import List
from promptolution.optimizers.base_optimizer import BaseOptimizer


class EvoPromptDE(BaseOptimizer):
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

        self._on_train_end()
        return self.prompts