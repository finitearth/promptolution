"""Module for LLM predictors."""

# Re-export the factory function from helpers
from ..helpers import get_predictor
from .base_predictor import BasePredictor
from .classifier import FirstOccurrenceClassifier, MarkerBasedClassifier

# Define what symbols are exported by default when using 'from promptolution.predictors import *'
__all__ = ["BasePredictor", "FirstOccurrenceClassifier", "MarkerBasedClassifier", "get_predictor"]
