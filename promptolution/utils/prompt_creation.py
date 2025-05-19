"""Utility functions for prompt creation."""


import numpy as np

from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from promptolution.llms.base_llm import BaseLLM
    from promptolution.tasks.base_task import BaseTask

from promptolution.optimizers.templates import (
    PROMPT_CREATION_TEMPLATE,
    PROMPT_CREATION_TEMPLATE_TD,
    PROMPT_VARIATION_TEMPLATE,
)
from promptolution.tasks.classification_tasks import ClassificationTask


def create_prompt_variation(prompt: Union[List[str], str], llm: "BaseLLM", meta_prompt: str = None) -> List[str]:
    """Generate a variation of the given prompt(s) while keeping the semantic meaning.

    Idea taken from the paper Zhou et al. (2021) https://arxiv.org/pdf/2211.01910

    Args:
        prompt (Union[List[str], str]): The prompt(s) to generate variations of.
        llm (BaseLLM): The language model to use for generating the variations.
        meta_prompt (str): The meta prompt to use for generating the variations.
        If None, a default meta prompt is used. Should contain <prev_prompt> tag.

    Returns:
        List[str]: A list of generated variations of the input prompt(s).
    """
    meta_prompt = PROMPT_VARIATION_TEMPLATE if meta_prompt is None else meta_prompt

    if isinstance(prompt, str):
        prompt = [prompt]
    varied_prompts = llm.get_response([meta_prompt.replace("<prev_prompt>", p) for p in prompt])

    varied_prompts = [p.split("</prompt>")[0].split("<prompt>")[-1] for p in varied_prompts]

    return varied_prompts


def create_prompts_from_samples(
    task: "BaseTask",
    llm: "BaseLLM",
    meta_prompt: str = None,
    n_samples: int = 3,
    task_description: str = None,
    n_prompts: int = 1,
    get_uniform_labels: bool = False,
) -> List[str]:
    """Generate a set of prompts from dataset examples sampled from a given task.

    Idea taken from the paper Zhou et al. (2021) https://arxiv.org/pdf/2211.01910
    Samples are selected, such that
    (1) all possible classes are represented
    (2) the samples are as representative as possible

    Args:
        task (BaseTask): The task to generate prompts for.
        Xs and Ys from this object are used to generate the prompts.
        llm (BaseLLM): The language model to use for generating the prompts.
        meta_prompt (str): The meta prompt to use for generating the prompts.
        If None, a default meta prompt is used.
        n_samples (int): The number of samples to use for generating prompts.
        task_description (str): The description of the task to include in the prompt.
        n_prompts (int): The number of prompts to generate.
        get_uniform_labels (bool): If True, samples are selected such that all classes are represented.

    Returns:
        List[str]: A list of generated prompts.
    """
    if meta_prompt is None and task_description is None:
        meta_prompt_template = PROMPT_CREATION_TEMPLATE
    elif meta_prompt is None and task_description is not None:
        meta_prompt_template = PROMPT_CREATION_TEMPLATE_TD.replace("<task_desc>", task_description)
    elif meta_prompt is not None and task_description is None:
        meta_prompt_template = meta_prompt
    elif meta_prompt is not None and task_description is not None:
        meta_prompt_template = meta_prompt.replace("<task_desc>", task_description)

    meta_prompts = []
    for _ in range(n_prompts):
        if isinstance(task, ClassificationTask) and get_uniform_labels:
            # if classification task sample such that all classes are represented
            unique_labels, counts = np.unique(task.ys, return_counts=True)
            proportions = counts / len(task.ys)
            samples_per_class = np.round(proportions * n_samples).astype(int)
            samples_per_class = np.maximum(samples_per_class, 1)

            # sample
            xs = []
            ys = []
            for label, n_samples in zip(unique_labels, samples_per_class):
                indices = np.where(task.ys == label)[0]
                indices = np.random.choice(indices, n_samples, replace=False)
                xs.extend(task.xs[indices])
                ys.extend(task.ys[indices])

        else:
            # if not classification task, sample randomly
            indices = np.random.choice(len(task.xs), n_samples, replace=False)
            xs = task.xs[indices].tolist()
            ys = task.ys[indices].tolist()

        examples = "\n\n".join([f"Input: {x}\nOutput: {y}" for x, y in zip(xs, ys)])
        meta_prompt = meta_prompt_template.replace("<input_output_pairs>", examples)
        meta_prompts.append(meta_prompt)

    prompts = llm.get_response(meta_prompts)
    prompts = [prompt.split("</prompt>")[0].split("<prompt>")[-1].strip() for prompt in prompts]

    return prompts
