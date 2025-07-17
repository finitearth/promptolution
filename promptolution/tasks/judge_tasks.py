"""Module for judge tasks."""

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Literal, Optional

from promptolution.llms.base_llm import BaseLLM
from promptolution.tasks.base_task import BaseTask
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.logging import get_logger

if TYPE_CHECKING:
    from promptolution.predictors.base_predictor import BasePredictor
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

JUDGE_PROMPT_WITHOUT_GROUND_TRUTH = """You are an expert evaluator. Judge the correctness and appropriateness of the response, for the given task.

Task:
{task}

Input:
{input}

Response:
{response}

Evaluate how well the response addresses the input for the given task. Consider correctness, relevance, and completeness.

Provide a score from -5 to +5 where:
- -5: Completely wrong/inappropriate
- 0: Partially addresses the task
- +5: Fully correct and appropriate

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
        config: "ExperimentConfig" = None,
    ):
        """Initialize the JudgeTask."""
        self.description = task_description
        super().__init__(
            df=df,
            x_column=x_column,
            y_column=y_column,
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
            prompt = JUDGE_PROMPT_WITH_GROUND_TRUTH.replace("{ground_truth}", str(y))
        else:
            prompt = JUDGE_PROMPT_WITHOUT_GROUND_TRUTH

        prompt = (
            prompt.replace("{task}", self.task_description)
            .replace("{input}", x)
            .replace("{prediction}", pred)
            .replace("{response}", pred)
        )
        return prompt

    def _calculate_score(self, x: np.ndarray, y: np.ndarray, pred: np.ndarray) -> float:
        """Calculate the score for a single prediction using the LLM judge."""
        judge_prompt = self._construct_judge_prompt(x, pred, y)
        judge_response = self.judge_llm.get_response(judge_prompt)[0]
        score = extract_from_tag(judge_response, "<final_score>", "</final_score>")
        try:
            score = float(score)
        except (ValueError, TypeError):
            logger.error(f"Failed to parse score from judge response: {judge_response}. Using 0.0 as default.")
            score = 0.0

        return score
