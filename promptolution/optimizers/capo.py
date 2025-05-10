"""Implementation of the CAPO (Cost-Aware Prompt Optimization) algorithm."""
import random
from itertools import compress
from logging import getLogger
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from promptolution.llms.base_llm import BaseLLM
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask
from promptolution.templates import (
    CAPO_CROSSOVER_TEMPLATE,
    CAPO_DOWNSTREAM_TEMPLATE,
    CAPO_FEWSHOT_TEMPLATE,
    CAPO_MUTATION_TEMPLATE,
)
from promptolution.utils.token_counter import get_token_counter


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


class CAPOptimizer(BaseOptimizer):
    """Optimizer that evolves prompt instructions using crossover, mutation, and racing based on evaluation scores and statistical tests."""

    def __init__(
        self,
        initial_prompts: List[str],
        task: BaseTask,
        meta_llm: BaseLLM,
        downstream_llm: BaseLLM,
        length_penalty: float,
        crossovers_per_iter: int,
        upper_shots: int,
        max_n_blocks_eval: int,
        test_statistic: Callable,
        df_few_shots: pd.DataFrame = None,
        shuffle_blocks_per_iter: bool = True,
        crossover_meta_prompt: str = None,
        mutation_meta_prompt: str = None,
        callbacks: List[Callable] = [],
        predictor: BasePredictor = None,
    ):
        """Initializes the CAPOptimizer with various parameters for prompt evolution.

        Args:
            initial_prompts (List[str]): Initial prompt instructions.
            task (BaseTask): The task instance containing dataset and description.
            df_few_shots (pd.DataFrame): DataFrame containing few-shot examples. If None, will pop 20 % of datapoints from task.
            meta_llm (BaseLLM): The meta language model for crossover/mutation.
            downstream_llm (BaseLLM): The downstream language model used for responses.
            length_penalty (float): Penalty factor for prompt length.
            crossovers_per_iter (int): Number of crossover operations per iteration.
            upper_shots (int): Maximum number of few-shot examples per prompt.
            p_few_shot_reasoning (float): Probability of generating llm-reasoning for few-shot examples, instead of simply using input-output pairs.
            n_trials_generation_reasoning (int): Number of trials to generate reasoning for few-shot examples.
            max_n_blocks_eval (int): Maximum number of evaluation blocks.
            test_statistic (Callable): Function to test significance between prompts.
                Inputs are (score_a, score_b, n_evals) and returns True if A is better.
            shuffle_blocks_per_iter (bool, optional): Whether to shuffle blocks each
                iteration. Defaults to True.
            crossover_meta_prompt (str, optional): Template for crossover instructions.
            mutation_meta_prompt (str, optional): Template for mutation instructions.
            callbacks (List[Callable], optional): Callbacks for optimizer events.
            predictor (BasePredictor, optional): Predictor to evaluate prompt
                performance.
        """
        super().__init__(initial_prompts, task, callbacks, predictor)
        self.df_few_shots = df_few_shots if df_few_shots is not None else task.pop_datapoints(frac=0.2)
        self.meta_llm = meta_llm
        self.downstream_llm = downstream_llm

        self.crossover_meta_prompt = crossover_meta_prompt or CAPO_CROSSOVER_TEMPLATE
        self.mutation_meta_prompt = mutation_meta_prompt or CAPO_MUTATION_TEMPLATE

        self.population_size = len(initial_prompts)
        self.crossovers_per_iter = crossovers_per_iter
        self.upper_shots = upper_shots
        self.max_n_blocks_eval = max_n_blocks_eval
        self.test_statistic = test_statistic

        self.shuffle_blocks_per_iter = shuffle_blocks_per_iter
        self.length_penalty = length_penalty
        self.token_counter = get_token_counter(downstream_llm)

        self.scores = np.empty(0)

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
        sample_inputs = few_shot_samples["input"].values
        sample_targets = few_shot_samples["target"].values
        few_shots = [
            CAPO_FEWSHOT_TEMPLATE.replace("<input>", i).replace(
                "<output>", f"{self.predictor.begin_marker}{t}{self.predictor.end_marker}"
            )
            for i, t in zip(sample_inputs, sample_targets)
        ]
        # Select partition of the examples to generate reasoning from downstream model
        preds, seqs = self.predictor.predict(
            instruction,
            sample_inputs,
            return_seq=True,
        )
        preds, seqs = preds.reshape(num_examples), seqs.reshape(num_examples)

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
                self.crossover_meta_prompt.replace("<mother>", mother.instruction_text)
                .replace("<father>", father.instruction_text)
                .replace("<task_desc>", self.task.description)
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
            self.mutation_meta_prompt.replace("<instruction>", prompt.instruction_text).replace(
                "<task_desc>", self.task.description
            )
            for prompt in offsprings
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
        if self.shuffle_blocks_per_iter:
            random.shuffle(self.task.blocks)

        block_scores = []
        for i, (block_id, _) in enumerate(self.task.blocks):
            # new_scores shape: (n_candidates, n_samples)
            new_scores = self.task.evaluate(
                [c.construct_prompt() for c in candidates], block_id, self.predictor, return_agg_scores=False
            )

            # subtract length penalty
            prompt_lengths = np.array([self.token_counter(c) for c in candidates])
            rel_prompt_lengths = prompt_lengths / self.max_prompt_length

            new_scores = new_scores - self.length_penalty * rel_prompt_lengths[:, None]
            block_scores.append(new_scores)
            scores = np.concatenate(block_scores, axis=1)

            # boolean matrix C_ij indicating if candidate j is better than candidate i
            comparison_matrix = np.array(
                [[self.test_statistic(other_score, score) for other_score in scores] for score in scores]
            )

            # Sum along rows to get number of better scores for each candidate
            n_better = np.sum(comparison_matrix, axis=1)

            # Create mask for survivors and filter candidates
            candidates = list(compress(candidates, n_better < k))
            block_scores = [bs[n_better < k] for bs in block_scores]

            if len(candidates) <= k or i == self.max_n_blocks_eval:
                break

        avg_scores = self.task.evaluate(
            [c.construct_prompt() for c in candidates]
        )  # TODO: make sure its getting all the evals!!
        order = np.argsort(-avg_scores)[:k]
        candidates = [candidates[i] for i in order]
        self.scores = avg_scores[order]

        return candidates

    def _pre_optimization_loop(self):
        self.prompt_objects = self._initialize_population(self.prompts)
        self.prompts = [p.construct_prompt() for p in self.prompt_objects]
        self.max_prompt_length = max(self.token_counter(p) for p in self.prompt_objects)

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
