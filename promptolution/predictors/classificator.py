from typing import List

import numpy as np

from promptolution.predictors.base_predictor import BasePredictor


class Classificator(BasePredictor):
    """
    A predictor class for classification tasks using language models.

    This class takes a language model and a list of classes, and provides a method
    to predict classes for given prompts and input data.

    Attributes:
        llm: The language model used for generating predictions.
        classes (List[str]): The list of valid class labels.

    Inherits from:
        BasePredictor: The base class for predictors in the promptolution library.
    """

    def __init__(self, llm, classes, *args, **kwargs):
        """
        Initialize the Classificator.

        Args:
            llm: The language model to use for predictions.
            classes (List[str]): The list of valid class labels.
        """
        self.llm = llm
        self.classes = classes

    def predict(
        self,
        prompts: List[str],
        xs: np.ndarray,
    ) -> np.ndarray:
        """
        Predict classes for given prompts and input data.

        This method generates predictions using the language model and then
        extracts the predicted class from the model's output.

        Args:
            prompts (List[str]): The list of prompts to use for prediction.
            xs (np.ndarray): The input data array.

        Returns:
            np.ndarray: A 2D array of predicted classes, with shape (len(prompts), len(xs)).

        Note:
            The method concatenates each prompt with each input data point,
            passes it to the language model, and then extracts the first word
            in the response that matches a class in self.classes.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        preds = self.llm.get_response([prompt + "\n" + x for prompt in prompts for x in xs])

        response = []
        for pred in preds:
            predicted_class = ""
            for word in pred.split(" "):
                word = "".join([c for c in word if c.isalpha()])
                if word in self.classes:
                    predicted_class = word
                    break

            response.append(predicted_class)

        response = np.array(response).reshape(len(prompts), len(xs))
        return response
