from dataclasses import dataclass
from configparser import ConfigParser
from typing import List


@dataclass
class Config:
    task_name: str = ""
    task_descriptions_path: str = ""
    n_steps: int = 0
    optimizer: str = ""
    meta_prompt_path: str = ""
    meta_llms: str = ""
    downstream_llm: str = ""
    logging_dir: str = ""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = ConfigParser()
        self.config.read(config_path)
        self._parse_config()

    def _parse_config(self):
        self.task_name = self.config["task"]["task_name"]
        self.task_descriptions_path = self.config["task"]["task_descriptions_path"]
        self.n_steps = int(self.config["task"]["steps"])
        self.optimizer = self.config["optimizer"]["name"]
        self.meta_prompt_path = self.config["optimizer"]["meta_prompt_path"]
        self.meta_llm = self.config["meta_llm"]["name"]
        self.downstream_llm = self.config["downstream_llm"]["name"]
        self.logging_dir = self.config["logging"]["dir"]
        
        if self.optimizer == "evopromptga":
            self.selection_mode = self.config["optimizer"]["selection_mode"]
        elif self.optimizer == "evopromptde":
            self.selection_mode = self.config["optimizer"]["donor_random"]

        if "local" in self.meta_llm:
            self.meta_bs = int(self.config["meta_llm"]["batch_size"])
            
        if "local" in self.downstream_llm:
            self.downstream_bs = int(self.config["downstream_llm"]["batch_size"])
