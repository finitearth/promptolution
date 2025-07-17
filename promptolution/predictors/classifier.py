"""Module for classification predictors."""


import numpy as np

from typing import TYPE_CHECKING, List

from promptolution.predictors.base_predictor import BasePredictor
from promptolution.utils.formatting import extract_from_tag

if TYPE_CHECKING:
    from promptolution.utils.config import ExperimentConfig


class FirstOccurrenceClassifier(BasePredictor):
    """A predictor class for classification tasks using language models.

    This class takes a language model and a list of classes, and provides a method
    to predict classes for given prompts and input data. The class labels are extracted
    by matching the words in the prediction with the list of valid class labels.
    The first occurrence of a valid class label in the prediction is used as the predicted class.
    If no valid class label is found, the first class label in the list is used as the default prediction.

    Attributes:
        llm: The language model used for generating predictions.
        classes (List[str]): The list of valid class labels.
        config (ExperimentConfig, optional): Configuration for the classifier, overriding defaults.

    Inherits from:
        BasePredictor: The base class for predictors in the promptolution library.
    """

    def __init__(self, llm, classes, config: "ExperimentConfig" = None):
        """Initialize the FirstOccurrenceClassifier.

        Args:
            llm: The language model to use for predictions.
            classes (List[str]): The list of valid class labels.
            config (ExperimentConfig, optional): Configuration for the classifier, overriding defaults.
        """
        assert all([c.islower() for c in classes]), "Class labels should be lowercase."
        self.classes = classes

        self.extraction_description = (
            f"The task is to classify the texts into one of those classes: {', '.join(classes)}."
            "The first occurrence of a valid class label in the prediction is used as the predicted class."
        )

        super().__init__(llm, config)

    def _extract_preds(self, preds: List[str]) -> np.ndarray:
        """Extract class labels from the predictions, based on the list of valid class labels.

        Args:
            preds: The raw predictions from the language model.
        """
        response = []
        for pred in preds:
            predicted_class = self.classes[0]  # use first class as default pred
            for word in pred.split():
                word = "".join([c for c in word if c.isalnum()]).lower()
                if word in self.classes:
                    predicted_class = word
                    break

            response.append(predicted_class)

        response = np.array(response)
        return response


class MarkerBasedClassifier(BasePredictor):
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

    def __init__(
        self,
        llm,
        classes=None,
        begin_marker="<final_answer>",
        end_marker="</final_answer>",
        config: "ExperimentConfig" = None,
    ):
        """Initialize the MarkerBasedClassifier.

        Args:
            llm: The language model to use for predictions.
            classes (List[str]): The list of valid class labels. If None, does not force any class.
            begin_marker (str): The marker to use for extracting the class label.
            end_marker (str): The marker to use for extracting the class label.
            config (ExperimentConfig, optional): Configuration for the classifier, overriding defaults.
        """
        self.classes = classes
        self.begin_marker = begin_marker
        self.end_marker = end_marker

        if classes is not None:
            assert all([c.islower() for c in classes]), "Class labels should be lowercase."

            self.extraction_description = (
                f"The task is to classify the texts into one of those classes: {','.join(classes)}."
                f"The class label is extracted from the text that are between these markers: {begin_marker} and {end_marker}."
            )
        else:
            self.extraction_description = f"The class label is extracted from the text that are between these markers: {begin_marker} and {end_marker}."

        super().__init__(llm, config)

    def _extract_preds(self, preds: List[str]) -> np.ndarray:
        """Extract class labels from the predictions, by extracting the text following the marker.

        Args:
            preds: The raw predictions from the language model.
        """
        response = []
        for pred in preds:
            pred = extract_from_tag(pred, self.begin_marker, self.end_marker).lower()
            if self.classes is not None and pred not in self.classes:
                pred = self.classes[0]

            response.append(pred)

        response = np.array(response)
        return response
