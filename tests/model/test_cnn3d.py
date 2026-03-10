"""Tests for CNN3D global forecaster model."""

import numpy as np
import pandas as pd
import pytest
from sktime.forecasting.base import ForecastingHorizon

from computer_vision.model.cnn3d import CNN3D


@pytest.fixture(autouse=True)
def setup_env():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    try:
        import tensorflow as tf

        tf.random.set_seed(42)
    except ImportError:
        pass


@pytest.fixture
def panel_data():
    """Create Panel type data (MultiIndex: entity, timestamp)."""
    dates = pd.date_range(start="2024-01-01", periods=192, freq="h")
    ids = ["entity_1", "entity_2"]

    data_list = []
    for entity_id in ids:
        values = (
            10
            + 5 * np.sin(np.arange(192) * 2 * np.pi / 24)
            + np.random.normal(0, 0.5, 192)
        )
        df = pd.DataFrame(
            {"value": values},
            index=pd.MultiIndex.from_product(
                [[entity_id], dates], names=["id", "timestamp"]
            ),
        )
        data_list.append(df)

    return pd.concat(data_list)


def test_fit_predict_panel_data(panel_data):
    """Test that CNN3D can fit on Panel type data."""
    forecaster = CNN3D(epochs=1, batch_size=16, random_state=42)
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    print("Panel data head:")
    print(panel_data.head())

    # Fit should not raise an exception
    forecaster.fit(y=panel_data, fh=fh)
    y_pred = forecaster.predict(fh=fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert hasattr(forecaster, "model_")


def test_predict_panel_data(panel_data):
    """Test that CNN3D can make predictions on Panel type data."""
    forecaster = CNN3D(epochs=1, batch_size=16, random_state=42)
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    # Fit the model
    forecaster.fit(y=first_entity, fh=fh)

    # Get second entity for prediction (global forecasting)
    second_entity = panel_data.loc[panel_data.index.get_level_values(0).unique()[1]]

    # Make predictions on the second entity
    y_pred = forecaster.predict(fh=fh, y=second_entity)

    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert not y_pred.isna().any()
