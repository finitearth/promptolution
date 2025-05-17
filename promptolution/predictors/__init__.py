"""Module for LLM predictors."""

from typing import Literal

from .base_predictor import DummyPredictor
from .classifier import FirstOccurrenceClassifier, MarkerBasedClassifier
