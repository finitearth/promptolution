from logging import getLogger, INFO
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
        df = pd.DataFrame({"step": [self.step]*len(optimizer.prompts), "prompt": optimizer.prompts, "score": optimizer.scores})
        df.to_csv(self.path, mode='a', header=False, index=False)

    def on_train_end(self, logs=None):
        pass

class BestPromptCallback(Callback):
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
    def __init__(self, total_steps):
        self.pbar = tqdm(total=total_steps)

    def on_step_end(self, optimizer):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()