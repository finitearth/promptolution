"""Implementation of the CAPO (Cost-Aware Prompt Optimization) algorithm."""
import random
from itertools import compress
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from promptolution.config import ExperimentConfig
from promptolution.llms.base_llm import BaseLLM
from promptolution.logging import get_logger
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask
from promptolution.templates import (
    CAPO_CROSSOVER_TEMPLATE,
    CAPO_DOWNSTREAM_TEMPLATE,
    CAPO_FEWSHOT_TEMPLATE,
    CAPO_MUTATION_TEMPLATE,
)
from promptolution.utils import TestStatistics, get_test_statistic_func, get_token_counter

logger = get_logger(__name__)


class CAPOPrompt:
    """Represents a prompt consisting of an instruction and few-shot examples."""

    def __init__(self, instruction_text: str, few_shots: List[str]):
        """Initializes the Prompt with an instruction and associated examples.

        Args:
            instruction_text (str): The instruction or prompt text.
            few_shots (List[str]): List of examples as string.
        """
        self.instruction_text = instruction_text.strip()
        self.few_shots = few_shots

    def construct_prompt(self) -> str:
        """Constructs the full prompt string by replacing placeholders in the template with the instruction and formatted examples.

        Returns:
            str: The constructed prompt string.
        """
        few_shot_str = "\n\n".join(self.few_shots).strip()
        prompt = (
            CAPO_DOWNSTREAM_TEMPLATE.replace("<instruction>", self.instruction_text)
            .replace("<few_shots>", few_shot_str)
            .replace("\n\n\n\n", "\n\n")  # replace extra newlines if no few shots are provided
            .strip()
        )
        return prompt

    def __str__(self):
        """Returns the string representation of the prompt."""
        return self.construct_prompt()


class CAPO(BaseOptimizer):
    """CAPO: Cost-Aware Prompt Optimization.

    This class implements an evolutionary algorithm for optimizing prompts in large language models
    by incorporating racing techniques and multi-objective optimization. It uses crossover, mutation,
    and racing based on evaluation scores and statistical tests to improve efficiency while balancing
    performance with prompt length. It is adapted from the paper "CAPO: Cost-Aware Prompt Optimization" by Zehle et al., 2025.
    """

    def __init__(
        self,
        predictor: BasePredictor,
        task: BaseTask,
        meta_llm: BaseLLM,
        initial_prompts: List[str] = None,
        crossovers_per_iter: int = 4,
        upper_shots: int = 5,
        max_n_blocks_eval: int = 10,
        test_statistic: TestStatistics = "paired_t_test",
        alpha: float = 0.2,
        length_penalty: float = 0.05,
        df_few_shots: pd.DataFrame = None,
        crossover_template: str = None,
        mutation_template: str = None,
        callbacks: List[Callable] = [],
        config: ExperimentConfig = None,
    ):
        """Initializes the CAPOptimizer with various parameters for prompt evolution.

        Args:
            predictor (BasePredictor): The predictor for evaluating prompt performance.
            task (BaseTask): The task instance containing dataset and description.
            meta_llm (BaseLLM): The meta language model for crossover/mutation.
            initial_prompts (List[str]): Initial prompt instructions.
            crossovers_per_iter (int): Number of crossover operations per iteration.
            upper_shots (int): Maximum number of few-shot examples per prompt.
            p_few_shot_reasoning (float): Probability of generating llm-reasoning for few-shot examples, instead of simply using input-output pairs.
            max_n_blocks_eval (int): Maximum number of evaluation blocks.
            test_statistic (TestStatistics): Statistical test to compare prompt performance. Default is "paired_t_test".
            alpha (float): Significance level for the statistical test.
            length_penalty (float): Penalty factor for prompt length.
            df_few_shots (pd.DataFrame): DataFrame containing few-shot examples. If None, will pop 10% of datapoints from task.
            crossover_template (str, optional): Template for crossover instructions.
            mutation_template (str, optional): Template for mutation instructions.
            callbacks (List[Callable], optional): Callbacks for optimizer events.
            config (ExperimentConfig, optional): Configuration for the optimizer.
        """
        self.meta_llm = meta_llm
        self.downstream_llm = predictor.llm

        self.crossover_template = crossover_template or CAPO_CROSSOVER_TEMPLATE
        self.mutation_template = mutation_template or CAPO_MUTATION_TEMPLATE

        self.crossovers_per_iter = crossovers_per_iter
        self.upper_shots = upper_shots
        self.max_n_blocks_eval = max_n_blocks_eval
        self.test_statistic = get_test_statistic_func(test_statistic)
        self.alpha = alpha

        self.length_penalty = length_penalty
        self.token_counter = get_token_counter(self.downstream_llm)

        self.scores = np.empty(0)
        super().__init__(predictor, task, initial_prompts, callbacks, config)
        self.df_few_shots = df_few_shots if df_few_shots is not None else task.pop_datapoints(frac=0.1)
        if self.max_n_blocks_eval > self.task.n_blocks:
            logger.warning(
                f"ℹ️ max_n_blocks_eval ({self.max_n_blocks_eval}) is larger than the number of blocks ({self.task.n_blocks})."
                f" Setting max_n_blocks_eval to {self.task.n_blocks}."
            )
            self.max_n_blocks_eval = self.task.n_blocks
        self.population_size = len(self.prompts)

        if hasattr(self.predictor, "begin_marker") and hasattr(self.predictor, "end_marker"):
            self.target_begin_marker = self.predictor.begin_marker
            self.target_end_marker = self.predictor.end_marker
        else:
            self.target_begin_marker = ""
            self.target_end_marker = ""

    def _initialize_population(self, initial_prompts: List[str]) -> List[CAPOPrompt]:
        """Initializes the population of Prompt objects from initial instructions.

        Args:
            initial_prompts (List[str]): List of initial prompt instructions.

        Returns:
            List[Prompt]: Initialized population of prompts with few-shot examples.
        """
        population = []
        for instruction_text in initial_prompts:
            num_examples = random.randint(0, self.upper_shots)
            few_shots = self._create_few_shot_examples(instruction_text, num_examples)
            population.append(CAPOPrompt(instruction_text, few_shots))

        return population

    def _create_few_shot_examples(self, instruction: str, num_examples: int) -> List[Tuple[str, str]]:
        if num_examples == 0:
            return []

        few_shot_samples = self.df_few_shots.sample(num_examples, replace=False)
        sample_inputs = few_shot_samples[self.task.x_column].values
        sample_targets = few_shot_samples[self.task.y_column].values
        few_shots = [
            CAPO_FEWSHOT_TEMPLATE.replace("<input>", i).replace(
                "<output>", f"{self.target_begin_marker}{t}{self.target_end_marker}"
            )
            for i, t in zip(sample_inputs, sample_targets)
        ]
        # Select partition of the examples to generate reasoning from downstream model
        preds, seqs = self.predictor.predict(
            [instruction] * num_examples,
            sample_inputs,
            return_seq=True,
        )

        # Check which predictions are correct and get a single one per example
        for j in range(num_examples):
            # Process and clean up the generated sequences
            seqs[j] = seqs[j].replace(sample_inputs[j], "").strip()
            # Check if the prediction is correct and add reasoning if so
            if preds[j] == sample_targets[j]:
                few_shots[j] = CAPO_FEWSHOT_TEMPLATE.replace("<input>", sample_inputs[j]).replace("<output>", seqs[j])

        return few_shots

    def _crossover(self, parents: List[CAPOPrompt]) -> List[CAPOPrompt]:
        """Performs crossover among parent prompts to generate offsprings.

        Args:
            parents (List[CAPOPrompt]): List of parent prompts.

        Returns:
            List[Prompt]: List of new offsprings after crossover.
        """
        crossover_prompts = []
        offspring_few_shots = []
        for _ in range(self.crossovers_per_iter):
            mother, father = random.sample(parents, 2)
            crossover_prompt = (
                self.crossover_template.replace("<mother>", mother.instruction_text)
                .replace("<father>", father.instruction_text)
                .strip()
            )
            # collect all crossover prompts then pass them bundled to the meta llm (speedup)
            crossover_prompts.append(crossover_prompt)
            combined_few_shots = mother.few_shots + father.few_shots
            num_few_shots = (len(mother.few_shots) + len(father.few_shots)) // 2
            offspring_few_shot = random.sample(combined_few_shots, num_few_shots)
            offspring_few_shots.append(offspring_few_shot)

        child_instructions = self.meta_llm.get_response(crossover_prompts)

        offsprings = []
        for instruction, examples in zip(child_instructions, offspring_few_shots):
            instruction = instruction.split("<prompt>")[-1].split("</prompt>")[0].strip()
            offsprings.append(CAPOPrompt(instruction, examples))

        return offsprings

    def _mutate(self, offsprings: List[CAPOPrompt]) -> List[CAPOPrompt]:
        """Apply mutation to offsprings to generate new candidate prompts.

        Args:
            offsprings (List[CAPOPrompt]): List of offsprings to mutate.

        Returns:
            List[Prompt]: List of mutated prompts.
        """
        # collect all mutation prompts then pass them bundled to the meta llm (speedup)
        mutation_prompts = [
            self.mutation_template.replace("<instruction>", prompt.instruction_text) for prompt in offsprings
        ]
        new_instructions = self.meta_llm.get_response(mutation_prompts)

        mutated = []
        for new_instruction, prompt in zip(new_instructions, offsprings):
            new_instruction = new_instruction.split("<prompt>")[-1].split("</prompt>")[0].strip()
            p = random.random()

            if p < 1 / 3 and len(prompt.few_shots) < self.upper_shots:  # add a random few shot
                new_few_shot = self._create_few_shot_examples(new_instruction, 1)
                new_few_shots = prompt.few_shots + new_few_shot
            if 1 / 3 <= p < 2 / 3 and len(prompt.few_shots) > 0:  # remove a random few shot
                new_few_shots = random.sample(prompt.few_shots, len(prompt.few_shots) - 1)
            else:  # do not change few shots, but shuffle
                new_few_shots = prompt.few_shots

            random.shuffle(new_few_shots)
            mutated.append(CAPOPrompt(new_instruction, new_few_shots))

        return mutated

    def _do_racing(self, candidates: List[CAPOPrompt], k: int) -> List[CAPOPrompt]:
        """Perform the racing (selection) phase by comparing candidates based on their evaluation scores using the provided test statistic.

        Args:
            candidates (List[CAPOPrompt]): List of candidate prompts.
            k (int): Number of survivors to retain.

        Returns:
            List[Prompt]: List of surviving prompts after racing.
        """
        self.task.reset_block_idx()
        block_scores = []
        i = 0
        while len(candidates) > k and i < self.max_n_blocks_eval:
            # new_scores shape: (n_candidates, n_samples)
            new_scores = self.task.evaluate(
                [c.construct_prompt() for c in candidates], self.predictor, return_agg_scores=False
            )

            # subtract length penalty
            prompt_lengths = np.array([self.token_counter(c.construct_prompt()) for c in candidates])
            rel_prompt_lengths = prompt_lengths / self.max_prompt_length

            new_scores = new_scores - self.length_penalty * rel_prompt_lengths[:, None]
            block_scores.append(new_scores)
            scores = np.concatenate(block_scores, axis=1)

            # boolean matrix C_ij indicating if candidate j is better than candidate i
            comparison_matrix = np.array(
                [[self.test_statistic(other_score, score, self.alpha) for other_score in scores] for score in scores]
            )

            # Sum along rows to get number of better scores for each candidate
            n_better = np.sum(comparison_matrix, axis=1)

            # Create mask for survivors and filter candidates
            candidates = list(compress(candidates, n_better < k))
            block_scores = [bs[n_better < k] for bs in block_scores]

            i += 1
            self.task.increment_block_idx()

        avg_scores = self.task.evaluate(
            [c.construct_prompt() for c in candidates], self.predictor, strategy="evaluated"
        )
        order = np.argsort(-avg_scores)[:k]
        candidates = [candidates[i] for i in order]
        self.scores = avg_scores[order]

        return candidates

    def _pre_optimization_loop(self):
        self.prompt_objects = self._initialize_population(self.prompts)
        self.prompts = [p.construct_prompt() for p in self.prompt_objects]
        self.max_prompt_length = max(self.token_counter(p) for p in self.prompts)
        self.task.reset_block_idx()

    def _step(self) -> List[str]:
        """Perform a single optimization step.

        Returns:
            List[str]: The optimized list of prompts after the step.
        """
        offsprings = self._crossover(self.prompt_objects)
        mutated = self._mutate(offsprings)
        combined = self.prompt_objects + mutated

        self.prompt_objects = self._do_racing(combined, self.population_size)
        self.prompts = [p.construct_prompt() for p in self.prompt_objects]

        return self.prompts
