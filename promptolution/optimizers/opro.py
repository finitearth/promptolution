import numpy as np

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.llms.base_llm import BaseLLM

from typing import List

class Opro(BaseOptimizer):
    """
    Opro: Optimization by PROmpting
    
    Proposed by the paper "Large Language Models as Optimizers" by Yang et. al: https://arxiv.org/abs/2309.03409
    
    """

    def __init__(self, llm: BaseLLM, n_samples: int = 2, **args):
        self.llm = llm
        self.n_samples = n_samples
        with open("../templates/opro_template.txt") as f:
            self.meta_prompt = f.readlines()
        
        self.meta_prompt = self.meta_prompt.replace("<task_description>", self.task.description)

        super().__init__(**args)
        self.prompts = []
        self.scores = []


    def _sample_examples(self):
        sample_x = np.random.choice(self.task.xs, self.n_samples)
        sample_y = np.random.choice(self.task.ys, self.n_samples)

        return "\n".join([f"Input: {x}\nOutput: {y}" for x, y in zip(sample_x, sample_y)])
    

    def _format_old_instructions(self):
        return "".join([f"Old instruction: {prompt}\nScore: {score}\n\n" for prompt, score in zip(self.prompts, self.scores)])

    def optimize(self, n_steps: int) -> List[str]:
        for _ in range(n_steps):
            examples = self._sample_examples()
            meta_prompt = (
                self.meta_prompt
                .replace("<old_instructions>", self.prompts)
                .replace("<examples>", examples)
            )

            prompt = self.llm.get_response(meta_prompt)
            score = self.task.evaluate(prompt, self.predictor)
            
            self.prompts.append(prompt)
            self.scores.append(score)

            self._on_step_end()

        # obtain best prompt
        best_prompt = self.prompts[self.scores.index(max(self.scores))]
        
        self._on_epoch_end()

        return best_prompt






            

        