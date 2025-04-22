"""Module for EvoPromptDE optimizer."""

from typing import List

import numpy as np

from promptolution.callbacks import BaseCallback
from promptolution.config import ExperimentConfig
from promptolution.llms.base_llm import BaseLLM
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask


class EvoPromptDE(BaseOptimizer):
    """EvoPromptDE: Differential Evolution-based Prompt Optimizer.

    This class implements a differential evolution algorithm for optimizing prompts in large language models.
    It is adapted from the paper "Connecting Large Language Models with Evolutionary Algorithms
    Yields Powerful Prompt Optimizers" by Guo et al., 2023.

    The optimizer uses a differential evolution strategy to generate new prompts from existing ones,
    with an option to use the current best prompt as a donor.

    Attributes:
        prompt_template (str): Template for generating meta-prompts during evolution.
        donor_random (bool): If False, uses the current best prompt as a donor; if True, uses a random prompt.
        meta_llm: Language model used for generating child prompts from meta-prompts.

    Args:
        prompt_template (str): Template for meta-prompts.
        meta_llm: Language model for child prompt generation.
        donor_random (bool, optional): Whether to use a random donor. Defaults to False.
        n_eval_samples (int, optional): Number of samples for evaluation. Defaults to 20.
        config (ExperimentConfig, optional): Configuration for the experiment.
    """

    def __init__(
        self,
        predictor: BasePredictor,
        task: BaseTask,
        prompt_template: str,
        meta_llm: BaseLLM,
        initial_prompts: List[str] = None,
        donor_random: bool = False,
        n_eval_samples: int = 20,
        callbacks: List[BaseCallback] = None,
        config: ExperimentConfig = None,
    ):
        """Initialize the EvoPromptDE optimizer."""
        self.prompt_template = prompt_template
        self.n_eval_samples = n_eval_samples
        self.donor_random = donor_random
        self.meta_llm = meta_llm
        super().__init__(
            predictor=predictor, task=task, initial_prompts=initial_prompts, callbacks=callbacks, config=config
        )

    def _pre_optimization_loop(self):
        self.scores = self.task.evaluate(self.prompts, self.predictor, subsample=True, n_samples=self.n_eval_samples)
        self.prompts = [prompt for _, prompt in sorted(zip(self.scores, self.prompts), reverse=True)]
        self.scores = sorted(self.scores, reverse=True)

    def _step(self) -> List[str]:
        """Perform the optimization process for a specified number of steps.

        This method iteratively improves the prompts using a differential evolution strategy.
        It evaluates prompts, generates new prompts using the DE algorithm, and replaces
        prompts if the new ones perform better.


        Returns:
            List[str]: The optimized list of prompts after all steps.
        """
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
        child_prompts = [prompt.split("<prompt>")[-1].split("</prompt>")[0].strip() for prompt in child_prompts]

        child_scores = self.task.evaluate(child_prompts, self.predictor, subsample=True, n_samples=self.n_eval_samples)

        for i in range(len(self.prompts)):
            if child_scores[i] > self.scores[i]:
                self.prompts[i] = child_prompts[i]
                self.scores[i] = child_scores[i]

        self.prompts = [prompt for _, prompt in sorted(zip(self.scores, self.prompts), reverse=True)]
        self.scores = sorted(self.scores, reverse=True)

        return self.prompts
