"""Module for classification predictors."""

from typing import List, Tuple

import numpy as np

from promptolution.predictors.base_predictor import BasePredictor


class FirstOccurrenceClassificator(BasePredictor):
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
        self.extraction_description = (
            f"The task is to classify the texts into one of those classes: {', '.join(classes)}."
            "The first occurrence of a valid class label in the prediction is used as the predicted class."
        )

    def _extract_preds(self, preds: List[str]) -> np.ndarray:
        """Extract class labels from the predictions, based on the list of valid class labels.

        Args:
            preds: The raw predictions from the language model.
        """
        response = []
        for pred in preds:
            predicted_class = self.classes[0]  # use first class as default pred
            for word in pred.split():
                word = "".join([c for c in word if c.isalnum()])
                if word in self.classes:
                    predicted_class = word
                    break

            response.append(predicted_class)

        response = np.array(response)
        return response


class MarkerBasedClassificator(BasePredictor):
    """A predictor class for classification tasks using language models.

    This class takes a language model and a list of classes, and provides a method
    to predict classes for given prompts and input data. The class labels are extracted.

    Attributes:
        llm: The language model used for generating predictions.
        classes (List[str]): The list of valid class labels.
        marker (str): The marker to use for extracting the class label.

    Inherits from:
        BasePredictor: The base class for predictors in the promptolution library.
    """

    def __init__(self, llm, classes=None, begin_marker="<final_answer>", end_marker="</final_answer>", *args, **kwargs):
        """Initialize the Classificator.

        Args:
            llm: The language model to use for predictions.
            classes (List[str]): The list of valid class labels. If None, does not force any class.
            begin_marker (str): The marker to use for extracting the class label.
            end_marker (str): The marker to use for extracting the class label.
            *args, **kwargs: Additional arguments for the BasePredictor.
        """
        super().__init__(llm)
        assert all([c.islower() for c in classes]), "Class labels should be lowercase."
        self.classes = classes
        self.begin_marker = begin_marker
        self.end_marker = end_marker

        if self.classes is not None:
            self.extraction_description = (
                f"The task is to classify the texts into one of those classes: {','.join(classes)}."
                f"The class label is extracted from the text that are between these markers: {begin_marker} and {end_marker}."
            )
        else:
            self.extraction_description = f"The class label is extracted from the text that are between these markers: {begin_marker} and {end_marker}."

    def _extract_preds(self, preds: List[str]) -> np.ndarray:
        """Extract class labels from the predictions, by extracting the text following the marker.

        Args:
            preds: The raw predictions from the language model.
        """
        response = []
        for pred in preds:
            pred = pred.split(self.begin_marker)[-1].split(self.end_marker)[0].strip().lower()
            if self.classes is not None and pred not in self.classes:
                pred = self.classes[0]

            response.append(pred)

        response = np.array(response)
        return response
