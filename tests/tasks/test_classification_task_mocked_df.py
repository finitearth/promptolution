import pytest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from promptolution.tasks.classification_tasks import ClassificationTask
from tests.mocks.mock_predictor import MockPredictor


@pytest.fixture
def mock_dataframe():
    """Create a mock DataFrame with controlled behavior."""
    # Create a mock DataFrame
    mock_df = MagicMock(spec=pd.DataFrame)
    
    # Set up the y_column with lowercase values
    mock_df.__getitem__.return_value.str.lower = MagicMock(return_value=mock_df.__getitem__.return_value)
    
    # Set up unique values for classes
    mock_df.__getitem__.return_value.unique.return_value = np.array(["positive", "negative", "neutral"])
    
    # Set up values for xs and ys
    mock_df.__getitem__.return_value.values = np.array(["Example 1", "Example 2", "Example 3"])
    
    return mock_df


def test_task_with_mocked_dataframe(mock_dataframe):
    """Test ClassificationTask with a mocked DataFrame."""
    # Initialize task with mocked DataFrame
    task = ClassificationTask(
        df=mock_dataframe,
        description="Mocked DataFrame test",
        x_column="x",
        y_column="y"
    )
    
    # Verify that the mocked data was used
    assert len(task.classes) == 3
    assert len(task.xs) == 3
    assert len(task.ys) == 3
    
    # Create mock predictor
    mock_predictor = MockPredictor(
        classes=["positive", "negative", "neutral"],
        predetermined_predictions={}
    )
    
    # Override predict to return controlled values
    mock_predictor.predict = MagicMock(return_value=np.array(["positive", "negative", "neutral"]))
    
    # Evaluate
    scores = task.evaluate(["Classify:"], mock_predictor, return_agg_scores=True)
    
    # Mock predictor.predict should have been called once
    mock_predictor.predict.assert_called_once()
    
    # Scores should be an array of length 1
    assert scores.shape == (1,)