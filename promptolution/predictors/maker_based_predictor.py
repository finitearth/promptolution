"""Module for the MarkerBasedPredictor."""

from typing import TYPE_CHECKING, List, Optional

from promptolution.predictors.base_predictor import BasePredictor
from promptolution.utils.formatting import extract_from_tag

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM


class MarkerBasedPredictor(BasePredictor):
    """A predictor class task using language models.

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
        llm: "BaseLLM",
        classes: Optional[List[str]] = None,
        begin_marker: str = "<final_answer>",
        end_marker: str = "</final_answer>",
    ) -> None:
        """Initialize the MarkerBasedPredictor.

        Args:
            llm: The language model to use for predictions.
            classes (List[str]): The list of valid class labels. If None, does not force any class.
            begin_marker (str): The marker to use for extracting the class label.
            end_marker (str): The marker to use for extracting the class label.
        """
        self.classes = classes
        self.begin_marker = begin_marker
        self.end_marker = end_marker

        if classes is not None:
            assert all([c.islower() for c in classes]), "Class labels should be lowercase."

            self.extraction_description = (
                f"The task is to classify the texts into one of those classes: {', '.join(classes)}."
                f"The class label is extracted from the text that are between these markers: {begin_marker} and {end_marker}."
            )
        else:
            self.extraction_description = f"The class label is extracted from the text that are between these markers: {begin_marker} and {end_marker}."

        super().__init__(llm)

    def _extract_preds(self, preds: List[str]) -> List[str]:
        """Extract class labels from the predictions, by extracting the text following the marker.

        Args:
            preds: The raw predictions from the language model.
        """
        result = []
        for pred in preds:
            pred = extract_from_tag(pred, self.begin_marker, self.end_marker).lower()
            if self.classes is not None and pred not in self.classes:
                pred = self.classes[0]

            result.append(pred)

        return result
