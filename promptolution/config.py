from configparser import ConfigParser
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration class for the promptolution library.

    This class handles loading and parsing of configuration settings,
    either from a config file or from keyword arguments.

    Attributes:
        task_name (str): Name of the task.
        ds_path (str): Path to the dataset.
        n_steps (int): Number of optimization steps.
        optimizer (str): Name of the optimizer to use.
        meta_prompt_path (str): Path to the meta prompt file.
        meta_llms (str): Name of the meta language model.
        downstream_llm (str): Name of the downstream language model.
        evaluation_llm (str): Name of the evaluation language model.
        init_pop_size (int): Initial population size. Defaults to 10.
        logging_dir (str): Directory for logging. Defaults to "logs/run.csv".
        experiment_name (str): Name of the experiment. Defaults to "experiment".
        include_task_desc (bool): Whether to include task description. Defaults to False.
        random_seed (int): Random seed for reproducibility. Defaults to 42.
    """
    task_name: str
    ds_path: str
    n_steps: int
    optimizer: str
    meta_prompt_path: str
    meta_llms: str
    downstream_llm: str
    evaluation_llm: str
    init_pop_size: int = 10
    logging_dir: str = "logs/run.csv"
    experiment_name: str = "experiment"
    include_task_desc: bool = False
    random_seed: int = 42

    def __init__(self, config_path: str = None, **kwargs):
        if config_path:
            self.config_path = config_path
            self.config = ConfigParser()
            self.config.read(config_path)
            self._parse_config()
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _parse_config(self):
        self.task_name = self.config["task"]["task_name"]
        self.ds_path = self.config["task"]["ds_path"]
        self.n_steps = int(self.config["task"]["steps"])
        self.random_seed = int(self.config["task"]["random_seed"])
        self.optimizer = self.config["optimizer"]["name"]
        self.meta_prompt_path = self.config["optimizer"]["meta_prompt_path"]
        self.meta_llm = self.config["meta_llm"]["name"]
        self.downstream_llm = self.config["downstream_llm"]["name"]
        self.evaluation_llm = self.config["evaluator_llm"]["name"]
        self.init_pop_size = int(self.config["optimizer"]["init_pop_size"])
        self.logging_dir = self.config["logging"]["dir"]
        self.experiment_name = self.config["experiment"]["name"]

        if "include_task_desc" in self.config["task"]:
            self.include_task_desc = self.config["task"]["include_task_desc"] == "True"

        if self.optimizer == "evopromptga":
            self.selection_mode = self.config["optimizer"]["selection_mode"]
        elif self.optimizer == "evopromptde":
            self.selection_mode = self.config["optimizer"]["donor_random"]

        if "local" in self.meta_llm:
            self.meta_bs = int(self.config["meta_llm"]["batch_size"])

        if "local" in self.downstream_llm:
            self.downstream_bs = int(self.config["downstream_llm"]["batch_size"])
