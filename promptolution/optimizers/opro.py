"""Module implementing the OPRO (Optimization by PROmpting) algorithm."""

from typing import Dict, List, Optional

import numpy as np

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
        meta_llm: BaseLLM,
        prompt_template: Optional[str] = None,
        max_num_instructions: int = 20,
        num_instructions_per_step: int = 8,
        num_few_shots: int = 3,
        **kwargs,
    ) -> None:
        """Initialize the OPRO optimizer.

        Args:
            df_few_shots: DataFrame with few-shot examples (must have 'input' and 'target' columns)
            meta_llm: LLM that generates improved prompts
            prompt_template: Custom meta prompt template (uses OPRO_TEMPLATE if None)
            max_num_instructions: Maximum previous instructions to include in meta prompt
            num_instructions_per_step: Number of prompts to generate in each step
            num_few_shots: Number of few-shot examples to include (0 for none)
            **kwargs: Additional arguments passed to the BaseOptimizer
        """
        super().__init__(**kwargs)
        self.meta_llm = meta_llm

        self.meta_prompt_template = prompt_template if prompt_template else OPRO_TEMPLATE
        self.max_num_instructions = max_num_instructions
        self.num_instructions_per_step = num_instructions_per_step
        self.num_few_shots = num_few_shots

    def _sample_examples(self) -> str:
        """Sample few-shot examples from the dataset.

        Returns:
            Formatted string of few-shot examples with inputs and expected outputs
        """
        idx = np.random.choice(len(self.task.xs), self.num_few_shots)
        sample_x = self.task.xs[idx]
        sample_y = self.task.ys[idx]

        return "\n".join([f"Input: {x}\nOutput: {y}" for x, y in zip(sample_x, sample_y)])

    def _format_instructions(self) -> str:
        """Format previous prompts and their scores for the meta prompt.

        Returns:
            Formatted string of previous prompts and their scores,
            sorted by ascending score (worse to better)
        """
        prompt_score_pairs = list(zip(self.prompts, self.scores))
        sorted_pairs = sorted(prompt_score_pairs, key=lambda x: x[1])

        return "".join([f"text:\n{prompt}\nscore: {int(100 * round(score, 2))}\n\n" for prompt, score in sorted_pairs])

    def _add_prompt_and_score(self, prompt: str, score: float) -> None:
        """Add a prompt and its score to the lists, maintaining max length.

        Args:
            prompt: The prompt to add
            score: The corresponding score for the prompt
        """
        if prompt in self.prompts:
            return

        self.prompts.append(prompt)
        self.scores.append(score)

        # Keep only the top-performing prompts if we exceed the maximum number of instructions
        keep_indices = np.argsort(self.scores)[-self.max_num_instructions :]
        self.prompts = [self.prompts[i] for i in keep_indices]
        self.scores = [self.scores[i] for i in keep_indices]

    def optimize(self, n_steps: int) -> List[str]:
        """Run the OPRO optimization process.

        Args:
            n_steps: Number of optimization steps to perform

        Returns:
            List of all prompts generated during optimization
        """
        self.scores = list(self.task.evaluate(self.prompts, self.predictor))
        self.meta_prompt = self.meta_prompt_template.replace("<instructions>", self._format_instructions()).replace(
            "<examples>", self._sample_examples()
        )

        for _ in range(n_steps):
            duplicate_prompts = 0
            for _ in range(self.num_instructions_per_step):
                generation_seed = np.random.randint(0, int(1e9))
                self.meta_llm.set_generation_seed(generation_seed)

                if self.verbosity > 1:
                    print(f"Seed: {generation_seed}")
                response = self.meta_llm.get_response([self.meta_prompt])[0]

                prompt = response.split("<prompt>")[-1].split("</prompt>")[0].strip()

                if prompt in self.prompts:
                    duplicate_prompts += 1
                    continue

                score = self.task.evaluate(prompt, self.predictor)[0]

                self._add_prompt_and_score(prompt, score)

                if self.verbosity > 1:
                    print(f"New Instruction: {prompt}\nScore: {score}\n")

            # Update meta prompt
            self.meta_prompt = self.meta_prompt_template.replace("<instructions>", self._format_instructions()).replace(
                "<examples>", self._sample_examples()
            )

            if self.verbosity > 1:
                print(f"New meta prompt:\n{self.meta_prompt}\n")

            continue_optimization = self._on_step_end()

            if not continue_optimization:
                break

            # stop optimization if all generated prompts are duplicates (converged)
            if duplicate_prompts == self.num_instructions_per_step:
                if self.verbosity > 0:
                    print("All generated prompts are duplicates. Stopping optimization.")
                break

        self._on_train_end()
        return self.prompts
