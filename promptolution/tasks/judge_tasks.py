"""Module for judge tasks."""

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, List, Literal, Optional

from promptolution.llms.base_llm import BaseLLM
from promptolution.tasks.base_task import BaseTask
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.utils.config import ExperimentConfig

logger = get_logger(__name__)

JUDGE_PROMPT_WITH_GROUND_TRUTH = """You are an expert evaluator. Judge how well the prediction matches the ground truth, for the given task.

Task:
{task}

Input:
{input}

Ground Truth:
{ground_truth}

Prediction:
{prediction}

Evaluate how closely the prediction aligns with the ground truth. Consider correctness, completeness, and accuracy of the match.

Provide a score from -5 to +5 where:
- -5: Completely incorrect/opposite
- 0: Partially correct
- +5: Perfect match

Return your answer encompased by <final_score></final_score>"""

JUDGE_PROMPT_WITHOUT_GROUND_TRUTH = """You are an expert evaluator. Judge the quality of the response, for the given task.

Task:
{task}

Input:
{input}

Prediction:
{prediction}

Evaluate how well the response addresses the input for the given task. Consider correctness, quality, relevance, completeness, and excellence of execution.

Provide a score from -5 to +5 where:
- -5: Completely wrong/inappropriate
- 0: Partially addresses the task with mixed quality
- +5: Exceptional response that brilliantly solves the task with creativity, insight, or outstanding execution that goes beyond basic correctness

Return your answer encompased by <final_score></final_score>"""


class JudgeTask(BaseTask):
    """Task that evaluates a predictor using an LLM as a judge, optionally accepting a ground truth."""

    def __init__(
        self,
        df: pd.DataFrame,
        judge_llm: "BaseLLM",
        x_column: str = "x",
        y_column: Optional[str] = None,
        task_description: Optional[str] = None,
        n_subsamples: int = 30,
        eval_strategy: Literal["full", "subsample", "sequential_block", "random_block"] = "full",
        seed: int = 42,
        judge_prompt: Optional[str] = None,
        min_score: float = -5.0,
        max_score: float = 5.0,
        config: "ExperimentConfig" = None,
    ):
        """Initialize the JudgeTask."""
        if judge_prompt is None:
            judge_prompt = JUDGE_PROMPT_WITH_GROUND_TRUTH if y_column else JUDGE_PROMPT_WITHOUT_GROUND_TRUTH
        self.judge_prompt = judge_prompt
        self.min_score = min_score
        self.max_score = max_score

        super().__init__(
            df=df,
            x_column=x_column,
            y_column=y_column,
            task_description=task_description,
            n_subsamples=n_subsamples,
            eval_strategy=eval_strategy,
            seed=seed,
            config=config,
        )
        assert judge_llm is not None, "judge_llm must be provided for JudgeTask"
        self.judge_llm = judge_llm

    def _construct_judge_prompt(self, x: str, pred: str, y: Optional[str] = None) -> str:
        """Constructs the judge prompt based on whether ground truth is available."""
        if y is not None:
            prompt = self.judge_prompt.replace("{ground_truth}", str(y))
        else:
            prompt = self.judge_prompt

        prompt = prompt.replace("{task}", self.task_description).replace("{input}", x).replace("{prediction}", pred)
        return prompt

    def _evaluate(self, xs: np.ndarray, ys: np.ndarray, preds: np.ndarray) -> List[float]:
        """Calculate the score for a single prediction using the LLM judge."""
        prompts: List[str] = []
        for x, y, pred in zip(xs, ys, preds):
            judge_prompt = self._construct_judge_prompt(x, pred, y)
            prompts.append(judge_prompt)
        judge_responses = self.judge_llm.get_response(prompts)
        scores_str = extract_from_tag(judge_responses, "<final_score>", "</final_score>")
        scores = []
        for score in scores_str:
            try:
                # only numeric chars, - or . are allowed
                score = "".join(filter(lambda c: c.isdigit() or c in "-.", score))
                score = float(score)
                # normalize from [-5, +5] to [0, 1]
                score = (score + self.min_score) / (self.max_score - self.min_score)
                score = max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Failed to parse score '{score}' as float. Defaulting to 0.0.")
                score = 0.0

            scores.append(score)

        return scores
