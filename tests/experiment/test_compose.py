"""Config composition + fail-fast tests for the experiment module."""

import pytest
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import MissingMandatoryValue

from tests.mocks.mock_llm import MockLLM


def test_defaults_compose(compose_cfg):
    cfg = compose_cfg()
    assert cfg.llm._target_.endswith("APILLM")
    assert cfg.optimizer._target_.endswith("CAPO")
    assert cfg.task._target_.endswith("ClassificationTask")
    assert cfg.n_steps == 10


def test_group_and_param_overrides(compose_cfg):
    cfg = compose_cfg(overrides=["optimizer=evopromptga", "n_steps=5"])
    assert cfg.optimizer._target_.endswith("EvoPromptGA")
    assert cfg.n_steps == 5


@pytest.mark.parametrize(
    "option, target",
    [("api_gpt-4o-mini", "APILLM"), ("vllm_qwen2.5-7b", "VLLM"), ("local_smollm2", "LocalLLM")],
)
def test_every_llm_backend_has_an_option(compose_cfg, option, target):
    """One example per backend, so all three LLM types are reachable from config."""
    cfg = compose_cfg(overrides=[f"llm={option}"])
    assert cfg.llm._target_.endswith(target)
    assert cfg.llm.model_id  # the option names a model, so it carries one


def test_llm_model_id_is_overridable(compose_cfg):
    """The options are examples: another model of the same backend needs no new file."""
    cfg = compose_cfg(overrides=["llm.model_id=gpt-4o"])
    assert cfg.llm.model_id == "gpt-4o"


def test_dataset_is_its_own_group(compose_cfg):
    """`df` is a top-level group, so the loader is swapped independently of the task type."""
    cfg = compose_cfg()
    assert cfg.df._target_ == "pandas.read_csv"
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.df.filepath_or_buffer  # no dataset ships with the package

    hf = compose_cfg(overrides=["df=huggingface_datasets", "df.path=SetFit/ag_news", "df.split=test"])
    assert hf.df._target_ == "datasets.load_dataset"


def test_task_requires_its_columns(compose_cfg):
    """The task says how to read the dataset, so those fields stay required."""
    cfg = compose_cfg()
    for key in ("x_column", "y_column", "task_description"):
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.task[key]


def test_task_exposes_subsampling_without_plus(compose_cfg):
    """n_subsamples and seed are declared, so they take a plain override."""
    cfg = compose_cfg(overrides=["task.n_subsamples=7"])
    assert cfg.task.n_subsamples == 7
    assert cfg.task.seed == cfg.random_seed  # seed is tied to the run seed


def test_meta_and_judge_llm_are_null_by_default(compose_cfg):
    cfg = compose_cfg()
    assert cfg.meta_llm is None  # falsy, so launch() shares the main llm
    assert cfg.judge_llm is None


@pytest.mark.parametrize("slot", ["meta_llm", "judge_llm"])
def test_llm_group_fills_a_second_slot(compose_cfg, slot):
    """`llm@<slot>=<option>` reuses the llm group without a duplicate group or a `+`."""
    cfg = compose_cfg(overrides=[f"llm@{slot}=vllm_qwen2.5-7b", f"{slot}.model_id=some-model"])
    assert cfg[slot]._target_.endswith("VLLM")
    assert cfg[slot].model_id == "some-model"


def test_judge_and_reward_task_options_compose(compose_cfg):
    judge = compose_cfg(overrides=["task=judge", "llm@judge_llm=api_gpt-4o-mini", "judge_llm.model_id=m"])
    assert judge.task._target_.endswith("JudgeTask")
    assert judge.task.judge_llm._target_.endswith("APILLM")  # pulled in from the llm group

    reward = compose_cfg(overrides=["task=reward", "task.reward_function._target_=mylib.rewards.r"])
    assert reward.task._target_.endswith("RewardTask")
    assert reward.task.reward_function._partial_ is True  # the function itself, not its result


def test_user_task_option_inherits_the_packaged_type(compose_cfg, tmp_path):
    """The documented workflow: define your dataset as your own option inheriting `classification`.

    Uses `defaults: - classification`, the relative sibling form. The absolute form
    (`- /task/classification`) silently nests the inherited node under `task.task` instead.
    """
    user_conf = tmp_path / "conf" / "task"
    user_conf.mkdir(parents=True)
    (user_conf / "mine.yaml").write_text(
        "defaults:\n"
        "  - classification\n"
        "  - _self_\n"
        "x_column: x\n"
        "y_column: y\n"
        'task_description: "Classify."\n'
    )
    cfg = compose_cfg(overrides=["task=mine", f"hydra.searchpath=[file://{tmp_path / 'conf'}]"])

    assert "task" not in cfg.task  # would be present if the inherited node had nested
    assert cfg.task._target_.endswith("ClassificationTask")  # inherited
    assert cfg.task.n_subsamples == 30  # inherited
    assert cfg.task.task_description == "Classify."  # overridden


def test_bad_param_fails_fast_at_instantiate(compose_cfg):
    """A misspelled param blows up at construction."""
    cfg = compose_cfg(overrides=["predictor=marker", "+predictor.begin_markerr=x"])
    # instantiate wraps the underlying TypeError; assert it raises and names the bad param
    with pytest.raises(InstantiationException, match="begin_markerr"):
        instantiate(cfg.predictor, llm=MockLLM())
