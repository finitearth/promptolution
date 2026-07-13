"""Tests for the one-call entry promptolution.optimize.optimize()."""

import json

import pandas as pd

from tests.mocks.mock_llm import MockLLM

from promptolution.optimize import optimize
from promptolution.optimizers.evoprompt_ga import EvoPromptGA

# One response string serves the meta roles (crossover/mutation, <prompt>) and downstream
# classification (<final_answer>) alike.
_RESPONSE = "<prompt>Classify the sentiment as positive or negative.</prompt><final_answer>positive</final_answer>"
# The initial-prompt generation expects a JSON list of 10 prompts as the first LLM call.
_INITIAL_PROMPTS_JSON = json.dumps([f"Classify the sentiment, variant {i}." for i in range(10)])


def _df(n=200):
    return pd.DataFrame({"x": [f"text number {i}" for i in range(n)], "y": (["positive", "negative"] * n)[:n]})


def _llm(first_responses=()):
    return MockLLM(predetermined_responses=list(first_responses) + [_RESPONSE] * 20000)


def test_optimize_defaults_end_to_end():
    """The headline UX: llm + data + task description in, scored prompts out."""
    llm = _llm(first_responses=[_INITIAL_PROMPTS_JSON])
    result = optimize(llm, _df(), task_description="Classify the sentiment as positive or negative.", n_steps=1)
    assert isinstance(result, pd.DataFrame) and {"prompt", "score"} <= set(result.columns)
    assert len(result) > 0
    assert result["score"].is_monotonic_decreasing  # best first


def test_optimize_optimizer_override():
    result = optimize(
        _llm(),
        _df(),
        task_description="Classify the sentiment as positive or negative.",
        optimizer=EvoPromptGA,
        initial_prompts=["Classify the sentiment.", "Positive or negative?"],
        n_steps=2,
    )
    assert len(result) == 2  # both prompts scored


def test_optimize_no_split_evaluates_on_train():
    result = optimize(
        _llm(),
        _df(),
        task_description="Classify the sentiment as positive or negative.",
        initial_prompts=["Classify the sentiment.", "Positive or negative?"],
        n_steps=1,
        test_frac=0,
    )
    assert len(result) == 2
