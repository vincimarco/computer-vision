from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections.abc

    import keras
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.compose import ColumnEnsembleTransformer, Id
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.impute import Imputer

from computer_vision.model.cnn3d import CNN3D
from computer_vision.transformer.cyclical_encoding import CyclicalEncodingTransformer


def create_forecaster(forecaster_name: str, params: dict) -> ForecastingPipeline:
    if forecaster_name == "cnn3d":
        return create_cnn3d_forecaster(**params)

    raise ValueError(f"Unknown forecaster: {forecaster_name}")


def create_cnn3d_forecaster(
    epochs: int,
    batch_size: int,
    random_state: int,
    loss: "str | keras.Metric",
    metrics: "collections.abc.Sequence[str | keras.Metric] | None",
    optimizer: "str | keras.optimizers.Optimizer",
    kernel_width: str,
    dropout_rate: float,
    sample_weights_function: "str | None",
    decay_rate: float,
    window_size: str,
) -> ForecastingPipeline:

    # _original_features_transformer = Id()

    _dtfeats_transformer = DateTimeFeatures() * ColumnEnsembleTransformer(
        [
            ("id", Id(), ["year"]),
            (
                "cyclical",
                CyclicalEncodingTransformer(),
                ["month_of_year", "day_of_week", "hour_of_day"],
            ),
        ],
        feature_names_out="original",
    )

    X_transformers = _dtfeats_transformer

    y_transformers = Imputer()

    forecaster = CNN3D(
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        kernel_width=kernel_width,
        dropout_rate=dropout_rate,
        sample_weights_function=sample_weights_function,
        decay_rate=decay_rate,
        window_size=window_size,
    )

    pipeline = X_transformers ** (y_transformers * forecaster)
    return pipeline
