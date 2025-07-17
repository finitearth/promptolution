"""Module for judge tasks."""

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Callable, Optional

from promptolution.tasks.base_task import BaseTask

if TYPE_CHECKING:
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.predictors.base_predictor import BasePredictor
    from promptolution.utils.config import ExperimentConfig

JUDGE_PROMPT_WITH_GROUND_TRUTH = """You are an expert evaluator. Judge how well the prediction matches the ground truth.

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

<final_score>score</final_score>"""

JUDGE_PROMPT_WITHOUT_GROUND_TRUTH = """You are an expert evaluator. Judge the correctness and appropriateness of the response.

Input:
{input}

Response:
{response}

Evaluate how well the response addresses the input for the given task. Consider correctness, relevance, and completeness.

Provide a score from -5 to +5 where:
- -5: Completely wrong/inappropriate
- 0: Partially addresses the task
- +5: Fully correct and appropriate

<final_score>score</final_score>"""


class JudgeTask(BaseTask):
    """Task that evaluates a predictor using an LLM as a judge, optionally accepting a ground truth."""

    def __init__(
        self,
        df: pd.DataFrame,
        x_column: str,
        judge: "BasePredictor",
        y_column: Optional[str] = None,
        config: "ExperimentConfig" = None,
    ):
        """Initialize the JudgeTask."""
        self.df = df
        self.x_column = x_column
        self.judge = judge
        self.y_column = y_column
        super().__init__(config)

    def _construct_judge_prompt(self, x, pred, y=None) -> str:
        if y:
            prompt = JUDGE_PROMPT_WITH_GROUND_TRUTH.replace("{ground_truth}", y)
        else:
            prompt = JUDGE_PROMPT_WITHOUT_GROUND_TRUTH

        prompt = prompt.replace("{input}", x).replace("{prediction}", pred)

        return prompt

    def evaluate(self, predictor: "BaseLLM", **kwargs) -> float:
        """Evaluate the predictor using the judge.

        Args:
            predictor (BasePredictor): The predictor to evaluate.
            **kwargs: Additional arguments for the judge.

        Returns:
            float: The mean score given by the judge.
        """
        inputs = self.df[self.x_column].tolist()
        predictions = predictor.predict(inputs, **kwargs)

        prompts = [
            self._construct_judge_prompt(x, pred, self.df[self.y_column][i] if self.y_column else None)
            for i, (x, pred) in enumerate(zip(inputs, predictions))
        ]

        scores = self.judge.get_response(prompts)
        scores = [float(score.split("<final_score>")[1].split("</final_score>")[0]) for score in scores]

        return np.mean(scores)
