"""Helper functions for the usage of the libary."""

from typing import List, Literal

import pandas as pd

from promptolution.config import ExperimentConfig
from promptolution.exemplar_selectors.random_search_selector import RandomSearchSelector
from promptolution.exemplar_selectors.random_selector import RandomSelector
from promptolution.llms.api_llm import APILLM
from promptolution.llms.base_llm import BaseLLM
from promptolution.llms.local_llm import LocalLLM
from promptolution.llms.vllm import VLLM
from promptolution.logging import get_logger
from promptolution.optimizers import CAPO
from promptolution.optimizers.evoprompt_de import EvoPromptDE
from promptolution.optimizers.evoprompt_ga import EvoPromptGA
from promptolution.optimizers.opro import Opro
from promptolution.predictors import FirstOccurrenceClassifier, MarkerBasedClassifier
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask
from promptolution.tasks.classification_tasks import ClassificationTask
from promptolution.templates import (
    CAPO_CROSSOVER_TEMPLATE,
    CAPO_MUTATION_TEMPLATE,
    EVOPROMPT_DE_TEMPLATE,
    EVOPROMPT_DE_TEMPLATE_TD,
    EVOPROMPT_GA_TEMPLATE,
    EVOPROMPT_GA_TEMPLATE_TD,
    OPRO_TEMPLATE,
    OPRO_TEMPLATE_TD,
)

logger = get_logger(__name__)


def run_experiment(df: pd.DataFrame, config: ExperimentConfig):
    """Run a full experiment based on the provided configuration.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        config (Config): Configuration object for the experiment.

    Returns:
        pd.DataFrame: A DataFrame containing the prompts and their scores.
    """
    # train test split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    prompts = run_optimization(train_df, config)
    df_prompt_scores = run_evaluation(test_df, config, prompts)

    return df_prompt_scores


def run_optimization(df, config: ExperimentConfig):
    """Run the optimization phase of the experiment.

    Args:
        config (Config): Configuration object for the experiment.

    Returns:
        List[str]: The optimized list of prompts.
    """
    llm = get_llm(config=config)
    predictor = get_predictor(llm, config=config)

    config.task_description = config.task_description + " " + predictor.extraction_description
    if config.optimizer == "capo" and config.subsample_strategy is None:
        logger.info("📌 CAPO requires block evaluation strategy. Setting it to 'sequential_block'.")
        config.subsample_strategy = "sequential_block"

    task = get_task(df, config)
    optimizer = get_optimizer(
        predictor=predictor,
        meta_llm=llm,
        task=task,
        config=config,
    )
    logger.info("🔥 Starting optimization...")
    prompts = optimizer.optimize(n_steps=config.n_steps)

    if hasattr(config, "prepend_exemplars") and config.prepend_exemplars:
        selector = get_exemplar_selector(config.exemplar_selector, task, predictor)
        prompts = [selector.select_exemplars(p, n_examples=config.n_exemplars) for p in prompts]

    return prompts


def run_evaluation(df: pd.DataFrame, config: ExperimentConfig, prompts: List[str]):
    """Run the evaluation phase of the experiment.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        config (Config): Configuration object for the experiment.
        prompts (List[str]): List of prompts to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing the prompts and their scores.
    """
    task = get_task(df, config)

    llm = get_llm(config=config)
    predictor = get_predictor(llm, config=config)
    logger.info("📊 Starting evaluation...")
    scores = task.evaluate(prompts, predictor)
    df = pd.DataFrame(dict(prompt=prompts, score=scores))
    df = df.sort_values("score", ascending=False, ignore_index=True)

    return df


def get_llm(model_id: str = None, config: ExperimentConfig = None):
    """Factory function to create and return a language model instance based on the provided model_id.

    This function supports three types of language models:
    1. LocalLLM: For running models locally.
    2. VLLM: For running models using the vLLM library.
    3. APILLM: For API-based models (default if not matching other types).

    Args:
        model_id (str): Identifier for the model to use. Special cases:
                        - "local-{model_name}" for LocalLLM
                        - "vllm-{model_name}" for VLLM
                        - Any other string for APILLM
        config (ExperimentConfig, optional): ExperimentConfig overwriting defaults.

    Returns:
        An instance of LocalLLM, or APILLM based on the model_id.
    """
    if model_id is None:
        model_id = config.llm
    if "local" in model_id:
        model_id = "-".join(model_id.split("-")[1:])
        return LocalLLM(model_id, config)
    if "vllm" in model_id:
        model_id = "-".join(model_id.split("-")[1:])
        return VLLM(model_id, config=config)

    return APILLM(model_id=model_id, config=config)


def get_task(df: pd.DataFrame, config: ExperimentConfig) -> BaseTask:
    """Get the task based on the provided DataFrame and configuration.

    So far only ClassificationTask is supported.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        config (ExperimentConfig): Configuration for the experiment.

    Returns:
        BaseTask: An instance of a task class based on the provided DataFrame and configuration.
    """
    return ClassificationTask(df, config=config)


def get_optimizer(
    predictor: BasePredictor,
    meta_llm: BaseLLM,
    task: BaseTask,
    optimizer: Literal["evopromptde", "evopromptga", "opro"] = None,
    meta_prompt: str = None,
    task_description: str = None,
    config: ExperimentConfig = None,
):
    """Creates and returns an optimizer instance based on provided parameters.

    Args:
        predictor: The predictor used for prompt evaluation
        meta_llm: The language model used for generating meta-prompts
        task: The task object used for evaluating prompts
        optimizer: String identifying which optimizer to use
        meta_prompt: Meta prompt text for the optimizer
        task_description: Description of the task for the optimizer
        config: Configuration object with default parameters

    Returns:
        An optimizer instance

    Raises:
        ValueError: If an unknown optimizer type is specified
    """
    if optimizer is None:
        optimizer = config.optimizer
    if task_description is None:
        task_description = config.task_description
    if meta_prompt is None and hasattr(config, "meta_prompt"):
        meta_prompt = config.meta_prompt

    if config.optimizer == "capo":
        crossover_template = (
            CAPO_CROSSOVER_TEMPLATE.replace("<task_desc>", task_description)
            if task_description
            else CAPO_CROSSOVER_TEMPLATE
        )
        mutation_template = (
            CAPO_MUTATION_TEMPLATE.replace("<task_desc>", task_description)
            if task_description
            else CAPO_MUTATION_TEMPLATE
        )

        return CAPO(
            predictor=predictor,
            meta_llm=meta_llm,
            task=task,
            crossover_template=crossover_template,
            mutation_template=mutation_template,
            config=config,
        )

    if config.optimizer == "evopromptde":
        template = (
            EVOPROMPT_DE_TEMPLATE_TD.replace("<task_desc>", task_description)
            if task_description
            else EVOPROMPT_DE_TEMPLATE
        )
        return EvoPromptDE(predictor=predictor, meta_llm=meta_llm, task=task, prompt_template=template, config=config)

    if config.optimizer == "evopromptga":
        template = (
            EVOPROMPT_GA_TEMPLATE_TD.replace("<task_desc>", task_description)
            if task_description
            else EVOPROMPT_GA_TEMPLATE
        )
        return EvoPromptGA(predictor=predictor, meta_llm=meta_llm, task=task, prompt_template=template, config=config)

    if config.optimizer == "opro":
        template = OPRO_TEMPLATE_TD.replace("<task_desc>", task_description) if task_description else OPRO_TEMPLATE
        return Opro(predictor=predictor, meta_llm=meta_llm, task=task, prompt_template=template, config=config)

    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_exemplar_selector(name: Literal["random", "random_search"], task: BaseTask, predictor: BasePredictor):
    """Factory function to get an exemplar selector based on the given name.

    Args:
        name (str): The name of the exemplar selector to instantiate.
        task (BaseTask): The task object to be passed to the selector.
        predictor (BasePredictor): The predictor object to be passed to the selector.

    Returns:
        BaseExemplarSelector: An instance of the requested exemplar selector.

    Raises:
        ValueError: If the requested selector name is not found.
    """
    if name == "random_search":
        return RandomSearchSelector(task, predictor)
    elif name == "random":
        return RandomSelector(task, predictor)
    else:
        raise ValueError(f"Unknown exemplar selector: {name}")


def get_predictor(downstream_llm=None, type: Literal["first_occurrence", "marker"] = "marker", *args, **kwargs):
    """Factory function to create and return a predictor instance.

    This function supports three types of predictors:
    1. FirstOccurrenceClassifier: A predictor that classifies based on first occurrence of the label.
    2. MarkerBasedClassifier: A predictor that classifies based on a marker.

    Args:
        downstream_llm: The language model to use for prediction.
        type (Literal["first_occurrence", "marker"]): The type of predictor to create:
                    - "first_occurrence" (default) for FirstOccurrenceClassifier
                    - "marker" for MarkerBasedClassifier
        *args: Variable length argument list passed to the predictor constructor.
        **kwargs: Arbitrary keyword arguments passed to the predictor constructor.

    Returns:
        An instance of FirstOccurrenceClassifier or MarkerBasedClassifier.
    """
    if type == "first_occurrence":
        return FirstOccurrenceClassifier(downstream_llm, *args, **kwargs)
    elif type == "marker":
        return MarkerBasedClassifier(downstream_llm, *args, **kwargs)
    else:
        raise ValueError(f"Invalid predictor type: '{type}'")
