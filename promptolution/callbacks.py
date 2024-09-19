import os

import pandas as pd
from tqdm import tqdm


class Callback:
    def on_step_end(self, optimizer):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class LoggerCallback(Callback):
    """
    Callback for logging optimization progress.

    This callback logs information about each step, epoch, and the end of training.

    Attributes:
        logger: The logger object to use for logging.
        step (int): The current step number.
    """
    def __init__(self, logger):
        # TODO check up whats up with logging leves
        self.logger = logger
        self.step = 0

    def on_step_end(self, optimizer):
        self.step += 1
        self.logger.critical(f"✨Step {self.step} ended✨")
        for i, (prompt, score) in enumerate(zip(optimizer.prompts, optimizer.scores)):
            self.logger.critical(f"*** Prompt {i}: Score: {score}")
            self.logger.critical(f"{prompt}")

    def on_epoch_end(self, epoch, logs=None):
        self.logger.critical(f"Epoch {epoch} - {logs}")

    def on_train_end(self, logs=None):
        self.logger.critical(f"Training ended - {logs}")


class CSVCallback(Callback):
    """
    Callback for saving optimization progress to a CSV file.

    This callback saves prompts and scores at each step to a CSV file.

    Attributes:
        path (str): The path to the CSV file.
        step (int): The current step number.
    """
    def __init__(self, path):
        # if dir does not exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # create file in path with header: "step,prompt,score"
        with open(path, "w") as f:
            f.write("step,prompt,score\n")
        self.path = path
        self.step = 0

    def on_step_end(self, optimizer):
        """
        Save prompts and scores to csv
        """
        self.step += 1
        df = pd.DataFrame(
            {"step": [self.step] * len(optimizer.prompts), "prompt": optimizer.prompts, "score": optimizer.scores}
        )
        df.to_csv(self.path, mode="a", header=False, index=False)

    def on_train_end(self, logs=None):
        pass


class BestPromptCallback(Callback):
    """
    Callback for tracking the best prompt during optimization.

    This callback keeps track of the prompt with the highest score.

    Attributes:
        best_prompt (str): The prompt with the highest score so far.
        best_score (float): The highest score achieved so far.
    """
    def __init__(self):
        self.best_prompt = ""
        self.best_score = -99999

    def on_step_end(self, optimizer):
        if optimizer.scores[0] > self.best_score:
            self.best_score = optimizer.scores[0]
            self.best_prompt = optimizer.prompts[0]

    def get_best_prompt(self):
        return self.best_prompt, self.best_score


class ProgressBarCallback(Callback):
    """
    Callback for displaying a progress bar during optimization.

    This callback uses tqdm to display a progress bar that updates at each step.

    Attributes:
        pbar (tqdm): The tqdm progress bar object.
    """
    def __init__(self, total_steps):
        self.pbar = tqdm(total=total_steps)

    def on_step_end(self, optimizer):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()
