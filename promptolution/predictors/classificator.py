"""Module for classification predictors."""

from typing import List

import numpy as np

from promptolution.predictors.base_predictor import BasePredictor


class Classificator(BasePredictor):
    """A predictor class for classification tasks using language models.

    This class takes a language model and a list of classes, and provides a method
    to predict classes for given prompts and input data.

    Attributes:
        llm: The language model used for generating predictions.
        classes (List[str]): The list of valid class labels.

    Inherits from:
        BasePredictor: The base class for predictors in the promptolution library.
    """

    def __init__(self, llm, classes, *args, **kwargs):
        """Initialize the Classificator.

        Args:
            llm: The language model to use for predictions.
            classes (List[str]): The list of valid class labels.
        """
        super().__init__(llm)
        self.classes = classes

    def _extract_preds(self, preds, shape):
        response = []
        for pred in preds:
            predicted_class = self.classes[0]  # use first class as default pred
            for word in pred.split(" "):
                word = "".join([c for c in word if c.isalnum()])
                if word in self.classes:
                    predicted_class = word
                    break

            response.append(predicted_class)

        response = np.array(response).reshape(*shape)
        return response
