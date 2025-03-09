"""Callback classes for logging, saving, and tracking optimization progress."""

import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


class Callback:
    """Base class for optimization callbacks."""

    def on_step_end(self, optimizer):
        """Called at the end of each optimization step.

        Args:
        optimizer: The optimizer object that called the callback.

        Returns:
            Bool: True if the optimization should continue, False if it should stop.
        """
        return True

    def on_epoch_end(self, optimizer):
        """Called at the end of each optimization epoch.

        Args:
        optimizer: The optimizer object that called the callback.

        Returns:
            Bool: True if the optimization should continue, False if it should stop.
        """
        return True

    def on_train_end(self, optimizer):
        """Called at the end of the entire optimization process.

        Args:
        optimizer: The optimizer object that called the callback.

        Returns:
            Bool: True if the optimization should continue, False if it should stop.
        """
        return True


class LoggerCallback(Callback):
    """Callback for logging optimization progress.

    This callback logs information about each step, epoch, and the end of training.

    Attributes:
        logger: The logger object to use for logging.
        step (int): The current step number.
    """

    def __init__(self, logger):
        """Initialize the LoggerCallback."""
        self.logger = logger
        self.step = 0

    def on_step_end(self, optimizer):
        """Log information about the current step."""
        self.step += 1
        self.logger.critical(f"✨Step {self.step} ended✨")
        for i, (prompt, score) in enumerate(zip(optimizer.prompts, optimizer.scores)):
            self.logger.critical(f"*** Prompt {i}: Score: {score}")
            self.logger.critical(f"{prompt}")

        return True

    def on_train_end(self, optimizer, logs=None):
        """Log information at the end of training.

        Args:
        optimizer: The optimizer object that called the callback.
        logs: Additional information to log.
        """
        self.logger.critical(f"Training ended - {logs}")

        return True


class CSVCallback(Callback):
    """Callback for saving optimization progress to a CSV file.

    This callback saves prompts and scores at each step to a CSV file.

    Attributes:
        dir (str): Directory the CSV file is saved to.
        step (int): The current step number.
    """

    def __init__(self, dir):
        """Initialize the CSVCallback.

        Args:
        dir (str): Directory the CSV file is saved to.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.dir = dir
        self.step = 0

    def on_step_end(self, optimizer):
        """Save prompts and scores to csv.

        Args:
        optimizer: The optimizer object that called the callback
        """
        self.step += 1
        df = pd.DataFrame(
            {
                "step": [self.step] * len(optimizer.prompts),
                "input_tokens": [optimizer.meta_llm.input_token_count] * len(optimizer.prompts),
                "output_tokens": [optimizer.meta_llm.output_token_count] * len(optimizer.prompts),
                "time_elapsed": [time.time() - optimizer.start_time] * len(optimizer.prompts),
                "score": optimizer.scores,
                "prompt": optimizer.prompts,
            }
        )

        if not os.path.exists(self.dir + "step_results.csv"):
            df.to_csv(self.dir + "step_results.csv", index=False)
        else:
            df.to_csv(self.dir + "step_results.csv", mode="a", header=False, index=False)

        return True

    def on_train_end(self, optimizer):
        """Called at the end of training.

        Args:
        optimizer: The optimizer object that called the callback.
        """
        df = pd.DataFrame(
            {
                "steps": self.step,
                "input_tokens": optimizer.meta_llm.input_token_count,
                "output_tokens": optimizer.meta_llm.output_token_count,
                "time_elapsed": time.time() - optimizer.start_time,
                "score": np.array(optimizer.scores).mean(),
                "best_prompts": str(optimizer.prompts),
            }
        )

        if not os.path.exists(self.dir + "train_results.csv"):
            df.to_csv(self.dir + "train_results.csv", index=False)
        else:
            df.to_csv(self.dir + "train_results.csv", mode="a", header=False, index=False)

        return True


class BestPromptCallback(Callback):
    """Callback for tracking the best prompt during optimization.

    This callback keeps track of the prompt with the highest score.

    Attributes:
        best_prompt (str): The prompt with the highest score so far.
        best_score (float): The highest score achieved so far.
    """

    def __init__(self):
        """Initialize the BestPromptCallback."""
        self.best_prompt = ""
        self.best_score = -99999

    def on_step_end(self, optimizer):
        """Update the best prompt and score if a new high score is achieved.

        Args:
        optimizer: The optimizer object that called the callback.
        """
        if optimizer.scores[0] > self.best_score:
            self.best_score = optimizer.scores[0]
            self.best_prompt = optimizer.prompts[0]

        return True

    def get_best_prompt(self):
        """Get the best prompt and score achieved during optimization.

        Returns:
        Tuple[str, float]: The best prompt and score.
        """
        return self.best_prompt, self.best_score


class ProgressBarCallback(Callback):
    """Callback for displaying a progress bar during optimization.

    This callback uses tqdm to display a progress bar that updates at each step.

    Attributes:
        pbar (tqdm): The tqdm progress bar object.
    """

    def __init__(self, total_steps):
        """Initialize the ProgressBarCallback.

        Args:
        total_steps (int): The total number of steps in the optimization process.
        """
        self.pbar = tqdm(total=total_steps)

    def on_step_end(self, optimizer):
        """Update the progress bar at the end of each step.

        Args:
        optimizer: The optimizer object that called the callback.
        """
        self.pbar.update(1)

        return True

    def on_train_end(self, optimizer):
        """Close the progress bar at the end of training.

        Args:
        optimizer: The optimizer object that called the callback.
        """
        self.pbar.close()

        return True


class TokenCountCallback(Callback):
    """Callback for stopping optimization based on the total token count."""

    def __init__(self, max_tokens_for_termination):
        """Initialize the TokenCountCallback."""
        self.max_tokens_for_termination = max_tokens_for_termination

    def on_step_end(self, optimizer):
        """Check if the total token count exceeds the maximum allowed. If so, stop the optimization."""
        token_counts = optimizer.predictor.llm.get_token_count()
        total_token_count = token_counts["total_tokens"]

        if total_token_count > self.max_tokens_for_termination:
            return False

        return True
