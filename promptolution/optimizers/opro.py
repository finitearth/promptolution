"""Module implementing the OPRO (Optimization by PROmpting) technique."""

from typing import Dict, List, Optional

import pandas as pd

from promptolution.llms.base_llm import BaseLLM
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.templates import OPRO_TEMPLATE


class Opro(BaseOptimizer):
    """OPRO: Optimization by PROmpting.

    Implementation of the technique proposed in "Large Language Models as Optimizers"
    (Yang et al., 2023: https://arxiv.org/abs/2309.03409).

    OPRO works by providing a meta-LLM with task descriptions and previous
    prompt-score pairs to generate improved prompts for a downstream LLM.
    """

    def __init__(
        self,
        df_few_shots: pd.DataFrame,
        downstream_llm: BaseLLM,
        meta_llm: BaseLLM,
        prompt_template: Optional[str] = None,
        n_eval_samples: int = 20,
        max_num_instructions: int = 20,
        num_instructions_per_step: int = 8,
        num_few_shots: int = 3,
        **kwargs,
    ) -> None:
        """Initialize the OPRO optimizer.

        Args:
            df_few_shots: DataFrame with few-shot examples (must have 'input' and 'target' columns)
            downstream_llm: LLM that will execute the optimized prompts
            meta_llm: LLM that generates improved prompts
            prompt_template: Custom meta prompt template (uses OPRO_TEMPLATE if None)
            n_eval_samples: Number of samples for evaluating each prompt
            max_num_instructions: Maximum previous instructions to include in meta prompt
            num_instructions_per_step: Number of prompts to generate in each step
            num_few_shots: Number of few-shot examples to include (0 for none)
            **kwargs: Additional arguments passed to the BaseOptimizer
        """
        super().__init__(**kwargs)
        self.df_few_shots = df_few_shots
        self.downstream_llm = downstream_llm
        self.meta_llm = meta_llm

        self.meta_prompt = prompt_template if prompt_template else OPRO_TEMPLATE

        if n_eval_samples <= 0:
            raise ValueError("n_eval_samples must be greater than 0")

        self.n_eval_samples = n_eval_samples
        self.max_num_instructions = max_num_instructions
        self.num_instructions_per_step = num_instructions_per_step
        self.num_few_shots = num_few_shots

        # Dictionary to store prompts and their scores
        self.prompt_score_dict: Dict[str, float] = {}

        # Initialize with existing prompts if any
        for prompt in self.prompts:
            if prompt not in self.prompt_score_dict:
                score = self.task.evaluate(prompt, self.predictor, subsample=True, n_samples=self.n_eval_samples)[0]
                self.prompt_score_dict[prompt] = score

    def _sample_examples(self) -> str:
        """Sample few-shot examples from the dataset.

        Returns:
            Formatted string of few-shot examples with inputs and expected outputs
        """
        if self.num_few_shots <= 0:
            return ""

        few_shot_samples = self.df_few_shots.sample(min(self.num_few_shots, len(self.df_few_shots)), replace=False)

        sample_inputs = few_shot_samples["input"].values
        sample_targets = few_shot_samples["target"].values

        return "\n".join([f"Input: {x}\nOutput: {y}" for x, y in zip(sample_inputs, sample_targets)])

    def _format_old_instructions(self) -> str:
        """Format previous prompts and their scores for the meta prompt.

        Returns:
            Formatted string of previous prompts and their scores,
            sorted by ascending score (worse to better)
        """
        sorted_instructions = sorted(self.prompt_score_dict.items(), key=lambda x: x[1])[: self.max_num_instructions]

        return "".join([f"text:\n{prompt}\nscore: {score}\n\n" for prompt, score in sorted_instructions])

    def optimize(self, n_steps: int) -> List[str]:
        """Run the OPRO optimization process.

        Args:
            n_steps: Number of optimization steps to perform

        Returns:
            List of all prompts generated during optimization
        """
        for _ in range(n_steps):
            meta_prompt = self.meta_prompt.replace("<old_instructions>", self._format_old_instructions()).replace(
                "<examples>", self._sample_examples()
            )

            for _ in range(self.num_instructions_per_step):
                response = self.meta_llm.get_response([meta_prompt])[0]

                prompt = response.split("<prompt>")[-1].split("</prompt>")[0].strip()

                if prompt in self.prompt_score_dict:
                    continue

                score = self.task.evaluate(prompt, self.predictor, subsample=True, n_samples=self.n_eval_samples)[0]

                self.prompt_score_dict[prompt] = score
                self.prompts.append(prompt)

            continue_optimization = self._on_step_end()
            if not continue_optimization:
                break

        self._on_epoch_end()

        self.scores = [self.prompt_score_dict[prompt] for prompt in self.prompts]

        return self.prompts
