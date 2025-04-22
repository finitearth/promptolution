"""Random exemplar selector."""

from promptolution.exemplar_selectors.base_exemplar_selector import BaseExemplarSelector
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask


class RandomSelector(BaseExemplarSelector):
    """A selector that randomly selects correct exemplars.

    This class implements a strategy that generates random examples and selects
    those that are evaluated as correct until the desired number of exemplars is reached.
    """

    def __init__(self, task: BaseTask, predictor: BasePredictor, desired_score: int = 1, config=None):
        """Initialize the RandomSelector.

        Args:
            task (BaseTask): An object representing the task to be performed.
            predictor (BasePredictor): An object capable of making predictions based on prompts.
            desired_score (int, optional): The desired score for the exemplars. Defaults to 1.
            config: ExperimentConfig overriding the defaults
        """
        self.desired_score = desired_score
        super().__init__(task, predictor, config)

    def select_exemplars(self, prompt, n_examples: int = 5):
        """Select exemplars using a random selection strategy.

        This method generates random examples and selects those that are evaluated as correct
        (score == self.desired_score) until the desired number of exemplars is reached.

        Args:
            prompt (str): The input prompt to base the exemplar selection on.
            n_examples (int, optional): The number of exemplars to select. Defaults to 5.

        Returns:
            str: A new prompt that includes the original prompt and the selected exemplars.
        """
        examples = []
        while len(examples) < n_examples:
            score, seq = self.task.evaluate(prompt, self.predictor, n_samples=1, return_seq=True)
            if score == self.desired_score:
                examples.append(seq[0])
        prompt = "\n\n".join([prompt] + examples) + "\n\n"

        return prompt
