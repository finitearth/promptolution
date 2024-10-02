import numpy as np

from typing import Union, List
from promptolution.llms.base_llm import BaseLLM
from promptolution.tasks.base_task import BaseTask

def create_prompt_variation(prompt: Union[List[str], str], llm: BaseLLM) -> List[str]:
    """
    Generate a variation of the given prompt(s) while keeping the semantic meaning. Idea taken from 
    the paper Zhou et al. (2021) https://arxiv.org/pdf/2211.01910

    Args:
        prompt (Union[List[str], str]): The prompt(s) to generate variations of.
        llm (BaseLLM): The language model to use for generating the variations.
    
    Returns:
        List[str]: A list of generated variations of the input prompt(s).
    """
    meta_prompt = """Generate a single variation of the following instruction while keeping the semantic meaning.
    Generate the variation starting with <prompt> and ending with </prompt> tags.
    
    Input: <prompt>
    
    Output:"""

    if isinstance(prompt, str):
        prompt = [prompt]
    varied_prompts = llm.get_response([meta_prompt.replace("<prompt>", p) for p in prompt])

    varied_prompts = [p.split("</prompt>")[0].split("<prompt>")[-1] for p in varied_prompts]

    return varied_prompts

def create_prompts_from_samples(task: BaseTask, llm: BaseLLM, n_samples: int = 3) -> List[str]:
    """
    Generate a set of prompts from dataset examples sampled from a given task. Idea taken from
    the paper Zhou et al. (2021) https://arxiv.org/pdf/2211.01910
    Samples are selected, such that 
    (1) all possible classes are represented
    (2) the samples are representative as possible

    Args:
        task (BaseTask): The task to generate prompts for. Xs and Ys from this object are used to generate the prompts.
        llm (BaseLLM): The language model to use for generating the prompts.
        n_samples (int): The number of samples to use for generating prompts.
    Returns:
        List[str]: A list of generated prompts.
    """
    unique_classes, counts = np.unique(task.ys, return_counts=True)
    proportions = counts / len(task.ys)
    samples_per_class = np.round(proportions * n_samples).astype(int)
    samples_per_class = np.maximum(samples_per_class, 1)

    # sample
    xs = []
    ys = []
    for cls, n_samples in zip(unique_classes, samples_per_class):
        indices = np.where(task.ys == cls)[0]
        indices = np.random.choice(indices, n_samples, replace=False)
        xs.extend(task.xs[indices])
        ys.extend(task.ys[indices])

    meta_prompt = """You are asked to give the corresponnding prompt that gives the following outputs given these inputs. Return it starting with <prompt> and ending with </prompt> tags. 
    Include the name of the output classes in the prompt."""
    for x, y in zip(xs, ys):
        meta_prompt += f"\n\nInput: {x}\nOutput: {y}"

    meta_prompt += "\nThe instruction was"

    prompt = llm.get_response([meta_prompt])[0]
    prompt = prompt.split("</prompt>")[0].split("<prompt>")[-1]


    return prompt

