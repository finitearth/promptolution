from promptolution.predictor import Predictor
from promptolution.tasks import ClassificationTask
from promptolution.optimizer import EvoPrompt

if __name__ == "__main__":
    intial_prompts = ["This is a classification task.", "This is a regression task."]
    optim = EvoPrompt()