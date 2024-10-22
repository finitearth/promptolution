"""Module for OPRO."""

from typing import List

import numpy as np

from promptolution.llms.base_llm import BaseLLM
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.templates import OPRO_TEMPLATE


class Opro(BaseOptimizer):
    """Opro: Optimization by PROmpting.

    Proposed by the paper "Large Language Models as Optimizers" by Yang et. al: https://arxiv.org/abs/2309.03409.
    This Optimizer works by providing the Meta-LLM with a task-description, as well as previous
    prompts with their respective score.

    Attributes:
        llm (BaseLLM): The Meta-LLM to optimize.
        n_samples (int): The number of samples from the task dataset to show the Meta-LLM.

    Methods:
        _sample_examples: Sample examples from the task dataset.
        _format_old_instructions: Format the previous prompts and their scores.
        optimize: Optimize the Meta-LLM by providing it with a new prompt.
    """

    def __init__(self, meta_llm: BaseLLM, n_samples: int = 2, prompt_template: str = None, **args):
        """Initialize the Opro optimizer."""
        self.meta_llm = meta_llm

        assert n_samples > 0, "n_samples must be greater than 0."
        self.n_samples = n_samples

        self.meta_prompt = prompt_template if prompt_template else OPRO_TEMPLATE

        super().__init__(**args)
        self.meta_prompt = self.meta_prompt.replace("<task_description>", self.task.description)

        self.scores = [
            self.task.evaluate(p, self.predictor, subsample=True, n_samples=self.n_eval_samples)[0] for p in self.prompts
        ]

    def _sample_examples(self):
        """Sample examples from the task dataset with their label.

        Returns:
            str: The formatted string of sampled examples.
        """
        idx = np.random.choice(len(self.task.xs), self.n_samples)
        sample_x = self.task.xs[idx]
        sample_y = self.task.ys[idx]

        return "\n".join([f"Input: {x}\nOutput: {y}" for x, y in zip(sample_x, sample_y)])

    def _format_old_instructions(self):
        """Format the previous prompts and their respective scores.

        Returns:
            str: The formatted string of previous prompts and their scores.
        """
        return "".join(
            [f"The old instruction was:\n{prompt}\nIt scored: {score}\n\n" for prompt, score in zip(self.prompts, self.scores)]
        )

    def optimize(self, n_steps: int) -> List[str]:
        """Optimize the Meta-LLM by providing it with a new prompt.

        Args:
            n_steps (int): The number of optimization steps to perform.

        Returns:
            str: The best prompt found by the optimizer.
        """
        for _ in range(n_steps):
            meta_prompt = self.meta_prompt.replace("<old_instructions>", self._format_old_instructions()).replace(
                "<examples>", self._sample_examples()
            )

            prompt = self.meta_llm.get_response([meta_prompt])[0]
            prompt = prompt.split("<prompt>")[-1].split("</prompt>")[0].strip()
            score = self.task.evaluate(prompt, self.predictor, subsample=True, n_samples=self.n_eval_samples)

            self.prompts.append(prompt)
            self.scores.append(score)

            self._on_step_end()

        self._on_epoch_end()

        return self.prompts
