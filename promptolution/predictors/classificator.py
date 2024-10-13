"""Module for classification predictors."""

<<<<<<< HEAD
from typing import List, Tuple
=======
from typing import List
>>>>>>> main

import numpy as np

from promptolution.predictors.base_predictor import BasePredictor


class Classificator(BasePredictor):
    """A predictor class for classification tasks using language models.

    This class takes a language model and a list of classes, and provides a method
    to predict classes for given prompts and input data. The class labels are extracted
    by matching the words in the prediction with the list of valid class labels.
    The first occurrence of a valid class label in the prediction is used as the predicted class.
    If no valid class label is found, the first class label in the list is used as the default prediction.

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

<<<<<<< HEAD
    def _extract_preds(self, preds: List[str], shape: Tuple[int, int]) -> np.ndarray:
        """Extract class labels from the predictions, based on the list of valid class labels.
=======
    def predict(
        self,
        prompts: List[str],
        xs: np.ndarray,
    ) -> np.ndarray:
        """Predict classes for given prompts and input data.

        This method generates predictions using the language model and then
        extracts the predicted class from the model's output.
>>>>>>> main

        Args:
            preds: The raw predictions from the language model.
            shape: The shape of the output array: (n_prompts, n_samples).
        """
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
