import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections.abc

    import keras
from sktime.forecasting.compose import ForecastingPipeline
from sktime.forecasting.darts import DartsXGBModel
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.transformations.compose import ColumnEnsembleTransformer, Id
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.impute import Imputer

from computer_vision.model.cnn3d import CNN3D
from computer_vision.transformer.cyclical_encoding import CyclicalEncodingTransformer


def create_forecaster(forecaster_name: str, params: dict) -> ForecastingPipeline:
    if forecaster_name == "cnn3d":
        return create_cnn3d_forecaster(**params)

    if forecaster_name == "naive":
        return create_naive_forecaster(**params)

    if forecaster_name == "darts_xgb":
        return create_darts_xgb_forecaster(**params)

    if forecaster_name == "lstm":
        return create_lstm_forecaster(**params)

    raise ValueError(f"Unknown forecaster: {forecaster_name}")


def load_forecaster(path: pathlib.Path) -> ForecastingPipeline:
    forecaster = ForecastingPipeline.load_from_path(path)
    return forecaster


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


def create_naive_forecaster(sp: int) -> ForecastingPipeline:
    forecaster = NaiveForecaster(strategy="last", sp=sp)
    pipeline = ForecastingPipeline([forecaster])
    return pipeline


def create_darts_xgb_forecaster(
    lags: int,
    output_chunk_length: int,
    random_state: int,
) -> ForecastingPipeline:

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
    forecaster = DartsXGBModel(
        lags=lags,
        output_chunk_length=output_chunk_length,
        random_state=random_state,
    )
    pipeline = X_transformers ** (y_transformers * forecaster)
    return pipeline


def create_lstm_forecaster(
    random_seed: int,
    loss: "str | keras.Metric",
    broadcasting: bool,
    optimizer: "str | keras.optimizers.Optimizer",
) -> ForecastingPipeline:

    import neuralforecast.losses.pytorch as nflosses
    import torch.optim

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

    losses = {
        "mse": nflosses.MSE(),
        "huber": nflosses.HuberLoss(),
    }

    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }

    forecaster = NeuralForecastLSTM(
        freq="15min",
        futr_exog_list=[
            "year",
            "month_of_year__sin",
            "month_of_year__cos",
            "day_of_week__sin",
            "day_of_week__cos",
            "hour_of_day__sin",
            "hour_of_day__cos",
        ],
        verbose_fit=True,
        verbose_predict=True,
        loss=losses[loss] if isinstance(loss, str) else loss,
        # optimizer=optimizers[optimizer] if isinstance(optimizer, str) else optimizer,
        random_seed=random_seed,
        broadcasting=broadcasting,
    )
    pipeline = X_transformers ** (y_transformers * forecaster)
    return pipeline
